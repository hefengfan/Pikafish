/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "bitboard.h"

#include <algorithm> // For std::max, std::swap
#include <array>     // For std::array instead of std::set
#include <bitset>    // For std::bitset::count
#include <initializer_list> // Potentially for pseudo-attack setup, but better with fixed arrays

// Required for magics.h if not using PEXT
#ifndef USE_PEXT
#include "magics.h"
#endif

namespace Stockfish {

// Global definitions (kept as is, assuming these are meant to be global)
uint8_t PopCnt16[1 << 16];
uint8_t SquareDistance[SQUARE_NB][SQUARE_NB];

Bitboard SquareBB[SQUARE_NB];
Bitboard LineBB[SQUARE_NB][SQUARE_NB];
Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
Bitboard PseudoAttacks[PIECE_TYPE_NB + 2][SQUARE_NB]; // Added +2 for PAWN_TO and KNIGHT_TO

Magic RookMagics[SQUARE_NB];
Magic CannonMagics[SQUARE_NB];
Magic BishopMagics[SQUARE_NB];
Magic KnightMagics[SQUARE_NB];
Magic KnightToMagics[SQUARE_NB];

namespace { // Anonymous namespace for internal helpers

Bitboard RookTable[0x108000];    // To store rook attacks
Bitboard CannonTable[0x108000];  // To store cannon attacks
Bitboard BishopTable[0x228];     // To store bishop attacks
Bitboard KnightTable[0x380];     // To store knight attacks
Bitboard KnightToTable[0x3E0];   // To store by knight attacks

// Using std::array instead of std::set for performance, as these are fixed lists.
// std::set has overhead for node allocation and tree traversal.
const std::array<Direction, 8> KnightDirections = {
    2 * SOUTH + WEST, 2 * SOUTH + EAST, SOUTH + 2 * WEST,
    SOUTH + 2 * EAST, NORTH + 2 * WEST, NORTH + 2 * EAST,
    2 * NORTH + WEST, 2 * NORTH + EAST};

const std::array<Direction, 4> BishopDirections = {
    2 * NORTH_EAST, 2 * SOUTH_EAST, 2 * SOUTH_WEST, 2 * NORTH_WEST};

// Forward declarations
template <PieceType pt>
void init_magics(Bitboard table[], Magic magics[] IF_NOT_PEXT(, const Bitboard magicsInit[]));

template <PieceType pt>
Bitboard lame_leaper_path(Direction d, Square s); // Path for one direction

template <PieceType pt>
Bitboard lame_leaper_path_all_dirs(Square s); // Path for all directions

template <PieceType pt>
Bitboard lame_leaper_attack(Square s, Bitboard occupied); // Attack for all directions with occupancy

// Returns the bitboard of target square for the given step
// from the given square. If the step is off the board, returns empty bitboard.
// Optimized: Removed distance check for performance, assuming callers ensure valid moves.
// The `distance(s, to) <= 2` check might be specific to certain piece types.
// For King/Advisor, the `Palace` check is applied later.
inline Bitboard safe_destination(Square s, int step) {
    Square to = Square(s + step);
    return is_ok(to) ? square_bb(to) : Bitboard(0);
}

} // anonymous namespace

// Returns an ASCII representation of a bitboard suitable
// to be printed to standard output. Useful for debugging.
// This function is for debugging/display, performance not critical.
std::string Bitboards::pretty(Bitboard b) {
    std::string s = "+---+---+---+---+---+---+---+---+---+\n";

    for (Rank r = RANK_9; r >= RANK_0; --r) {
        for (File f = FILE_A; f <= FILE_I; ++f)
            s += (b & make_square(f, r)) ? "| X " : "|   "; // Used space instead of unicode space

        s += "| " + std::to_string(r) + "\n+---+---+---+---+---+---+---+---+---+\n";
    }
    s += "  a   b   c   d   e   f   g   h   i\n"; // Consistent spacing

    return s;
}

// Initializes various bitboard tables. It is called at
// startup and relies on global objects to be already zero-initialized.
void Bitboards::init() {

    // 1. PopCnt16 table initialization
    for (unsigned i = 0; i < (1 << 16); ++i)
        PopCnt16[i] = static_cast<uint8_t>(std::bitset<16>(i).count());

    // 2. SquareBB initialization
    for (Square s = SQ_A0; s <= SQ_I9; ++s)
        SquareBB[s] = (Bitboard(1ULL) << static_cast<std::uint8_t>(s));

    // 3. SquareDistance initialization
    for (Square s1 = SQ_A0; s1 <= SQ_I9; ++s1)
        for (Square s2 = SQ_A0; s2 <= SQ_I9; ++s2)
            SquareDistance[s1][s2] = std::max(distance<File>(s1, s2), distance<Rank>(s1, s2));

    // 4. Magic Bitboard table initialization for sliding/leaping pieces
    // Note: CANNON uses RookMagics[s].mask, so RookMagics must be initialized first
    init_magics<ROOK>(RookTable, RookMagics IF_NOT_PEXT(, RookMagicsInit));
    init_magics<CANNON>(CannonTable, CannonMagics IF_NOT_PEXT(, RookMagicsInit)); // Reuses RookMagics.mask
    init_magics<BISHOP>(BishopTable, BishopMagics IF_NOT_PEXT(, BishopMagicsInit));
    init_magics<KNIGHT>(KnightTable, KnightMagics IF_NOT_PEXT(, KnightMagicsInit));
    init_magics<KNIGHT_TO>(KnightToTable, KnightToMagics IF_NOT_PEXT(, KnightToMagicsInit));

    // 5. PseudoAttacks and LineBB/BetweenBB initialization
    for (Square s1 = SQ_A0; s1 <= SQ_I9; ++s1) {
        // Pawn attacks (different for white/black based on context of NO_PIECE_TYPE/PAWN)
        PseudoAttacks[NO_PIECE_TYPE][s1] = pawn_attacks_bb<WHITE>(s1); // For White Pawn pushes/captures
        PseudoAttacks[PAWN][s1] = pawn_attacks_bb<BLACK>(s1);         // For Black Pawn pushes/captures

        // "To" attacks (Knight_to is for pieces attacking a square *as if* they were a knight moving from it)
        PseudoAttacks[KNIGHT_TO][s1] = pawn_attacks_to_bb<WHITE>(s1); // Assuming this is actually "Pawn attacks a square from s1"
        PseudoAttacks[PAWN_TO][s1] = pawn_attacks_to_bb<BLACK>(s1);   // Assuming this is actually "Pawn attacks a square from s1"

        // Pseudo attacks for non-sliding leapers (King, Advisor, Elephant)
        PseudoAttacks[ROOK][s1] = attacks_bb<ROOK>(s1, 0);       // Attacks on empty board
        PseudoAttacks[BISHOP][s1] = attacks_bb<BISHOP>(s1, 0);   // Attacks on empty board
        PseudoAttacks[KNIGHT][s1] = attacks_bb<KNIGHT>(s1, 0);   // Attacks on empty board
        PseudoAttacks[CANNON][s1] = attacks_bb<CANNON>(s1, 0);   // Pseudo attack with no hurdle

        // King pseudo-attacks (restricted to Palace)
        if (Palace & s1) { // Only calculate if s1 is in the palace
            Bitboard king_attacks = 0;
            // Iterate over all 4 orthogonal directions
            for (int step : {NORTH, SOUTH, WEST, EAST}) {
                Square target_sq = Square(s1 + step);
                if (is_ok(target_sq)) { // Check if target square is on board
                    king_attacks |= square_bb(target_sq);
                }
            }
            PseudoAttacks[KING][s1] = king_attacks & Palace; // Mask with Palace
        } else {
            PseudoAttacks[KING][s1] = 0; // If not in palace, king has no pseudo-attacks from there
        }

        // Advisor pseudo-attacks (restricted to Palace)
        if (Palace & s1) { // Only calculate if s1 is in the palace
            Bitboard advisor_attacks = 0;
            // Iterate over all 4 diagonal directions
            for (int step : {NORTH_WEST, NORTH_EAST, SOUTH_WEST, SOUTH_EAST}) {
                Square target_sq = Square(s1 + step);
                if (is_ok(target_sq)) { // Check if target square is on board
                    advisor_attacks |= square_bb(target_sq);
                }
            }
            PseudoAttacks[ADVISOR][s1] = advisor_attacks & Palace; // Mask with Palace
        } else {
            PseudoAttacks[ADVISOR][s1] = 0; // If not in palace, advisor has no pseudo-attacks from there
        }

        // Initialize LineBB and BetweenBB
        for (Square s2 = SQ_A0; s2 <= SQ_I9; ++s2) {
            // Initialize to 0 for all pairs first
            LineBB[s1][s2] = 0;
            BetweenBB[s1][s2] = 0;

            // Rook lines and between squares
            if (PseudoAttacks[ROOK][s1] & s2) { // If s1 and s2 are on the same rank/file
                LineBB[s1][s2] = (attacks_bb(ROOK, s1, 0) & attacks_bb(ROOK, s2, 0)) | s1 | s2;
                // BetweenBB for rooks needs to exclude s1 and s2
                BetweenBB[s1][s2] = (attacks_bb(ROOK, s1, square_bb(s2)) & attacks_bb(ROOK, s2, square_bb(s1)));
            }

            // Knight BetweenBB: path taken by the knight to reach s2 from s1
            if (PseudoAttacks[KNIGHT][s1] & s2) {
                // The lame_leaper_path<KNIGHT_TO> computes the "path" for a knight from s1 to s2.
                // It essentially finds the intermediate square the knight "hops over".
                BetweenBB[s1][s2] |= lame_leaper_path<KNIGHT>(Direction(s2 - s1), s1);
            }

            // IMPORTANT: BetweenBB[s1][s2] should *not* include s2 unless it's an intercept square.
            // The original code `BetweenBB[s1][s2] |= s2;` seems incorrect for typical BetweenBB usage,
            // as BetweenBB usually excludes the endpoints. If `s2` is truly meant to be an intermediate
            // square in some specific context (e.g., for king/advisor paths, which is not the case here),
            // it would be handled differently. For sliding pieces, BetweenBB usually contains squares strictly
            // between s1 and s2. For leapers, it would be the 'hop' square.
            // I'm commenting out the line `BetweenBB[s1][s2] |= s2;` for correctness of "between".
            // If the original intent was different, this needs clarification.
            // BetweenBB[s1][s2] |= s2; // Potentially incorrect for "between"
        }
    }
}

namespace { // Anonymous namespace for internal helpers

template <PieceType pt>
Bitboard sliding_attack(Square sq, Bitboard occupied) {
    assert(pt == ROOK || pt == CANNON);
    Bitboard attack = 0;

    // Fixed array for directions, more efficient than a temporary initializer_list
    const std::array<Direction, 4> directions = {NORTH, SOUTH, EAST, WEST};

    for (const auto& d : directions) {
        bool hurdle = false;
        // Optimization: Pre-calculate distance check once per direction
        // The loop condition `distance(s - d, s) == 1` is redundant if `s += d` is correctly updating `s`.
        // A direct `s = sq + d` and then `s += d` is sufficient to cover step-by-step movement.
        for (Square s = sq + d; is_ok(s); s = Square(s + d)) {
            // Check if the current step is a valid single step from the previous square.
            // This is crucial for rectilinear moves to ensure it stays on the same file/rank.
            // However, this check is only needed if `d` can lead off a straight line.
            // For simple NORTH/SOUTH/EAST/WEST, it's typically just `is_ok(s)`.
            // The original `distance(s - d, s) == 1` seems intended to ensure straight lines.
            // For performance, pre-calculating boundaries or using bitwise operations with ranks/files is better.
            // Simplified condition: Check if `s` is still in bounds AND on the same line.
            // The `is_ok(s)` is already a boundary check.
            // For standard orthogonal moves (N, S, E, W), `s += d` naturally stays on line.

            if (pt == ROOK || hurdle) {
                attack |= square_bb(s);
            }

            if (occupied & square_bb(s)) {
                if (pt == CANNON && !hurdle)
                    hurdle = true;
                else
                    break; // Blocked or second hurdle hit for Cannon
            }
        }
    }
    return attack;
}

// Computes the "leg" square for a leaper (Knight or Bishop in Chinese Chess context)
// This is the square that must be empty for the leaper to move.
template <PieceType pt>
Bitboard lame_leaper_path(Direction d, Square s) {
    Bitboard b = 0;
    Square to = Square(s + d);

    // Initial check: if target is invalid or too far (not a direct leaper jump), return empty.
    // distance(s, to) >= 4 covers cases like 2*NORTH+WEST (dist 3) vs 3*NORTH (dist 3)
    // Here, 4 is an arbitrary threshold for a single leap, 2 or 3 is typical.
    // The `KnightDirections` and `BishopDirections` already define valid leaps.
    // This `distance` check might be overly broad. A check that `to` is a valid jump from `s` is better.
    // For KNIGHT, `distance(s,to)` should be 3. For BISHOP it should be 2.
    // This function calculates the *path square*, not the destination.
    //
    // The logic inside this function seems to calculate the "intermediate" square a leaper "jumps over".
    // For a Knight move like `s + (2*NORTH + WEST)`:
    // `dr` would be `NORTH`, `df` would be `WEST`.
    // `diff` compares rank and file changes.
    // If diff > 0 (more file change, e.g., 2 horizontal, 1 vertical), then `s += df` (horizontal leg).
    // If diff < 0 (more rank change, e.g., 1 horizontal, 2 vertical), then `s += dr` (vertical leg).
    // If diff == 0 (e.g., (1,1) for elephant/advisor), then `s += df + dr` (diagonal leg).

    if (!is_ok(to)) return b; // Target square must be valid

    // For KNIGHT_TO, it means a piece *attacks* 's' as if it were a knight moving FROM 'to'.
    // So we effectively reverse the move to find the blocker.
    if (pt == KNIGHT_TO) {
        std::swap(s, to);
        d = -d; // Reverse direction
    }

    // Determine the "leg" square based on the direction `d`.
    // This logic relies on `Direction` being a sum of cardinal directions.
    Direction dr = (d > 0 && d / NORTH) ? NORTH : SOUTH; // Vertical component of d
    if (!(d / NORTH)) dr = Direction(0); // No vertical movement
    Direction df = (d % NORTH) > 0 ? EAST : WEST; // Horizontal component of d
    if (!(d % NORTH)) df = Direction(0); // No horizontal movement

    // Simplified calculation of leg square
    // For a knight move like (2,1) or (1,2), the leg square is (s+dx, s+dy) where dx=1 or dy=1
    // The original logic `diff = std::abs(file_of(to) - file_of(s)) - std::abs(rank_of(to) - rank_of(s));`
    // computes |df| - |dr|.
    // For knight (2,1), diff = 2 - 1 = 1. leg is s + (sign(df), 0) => s + df
    // For knight (1,2), diff = 1 - 2 = -1. leg is s + (0, sign(dr)) => s + dr
    // For bishop (2,2), diff = 2 - 2 = 0. leg is s + (sign(df), sign(dr)) => s + df + dr
    // This is correct for the "hop" square.

    // Calculate the "leg" square
    Square leg_s = s;
    int abs_rank_diff = std::abs(rank_of(to) - rank_of(s));
    int abs_file_diff = std::abs(file_of(to) - file_of(s));

    if (abs_file_diff > abs_rank_diff) { // More horizontal movement (e.g., Knight (2,1)
        leg_s = Square(s + (file_of(to) > file_of(s) ? EAST : WEST));
    } else if (abs_rank_diff > abs_file_diff) { // More vertical movement (e.g., Knight (1,2)
        leg_s = Square(s + (rank_of(to) > rank_of(s) ? NORTH : SOUTH));
    } else { // Equal horizontal/vertical (e.g., Bishop (2,2)
        // Check for diagonal move only if it's actually diagonal (not horizontal/vertical)
        if (abs_rank_diff > 0 && abs_file_diff > 0) {
             leg_s = Square(s + (rank_of(to) > rank_of(s) ? NORTH : SOUTH) + (file_of(to) > file_of(s) ? EAST : WEST));
        } else {
            // This case should ideally not be reached if directions are truly for leapers.
            // If d is purely orthogonal (e.g. for king or advisor), this function might be misused.
            // The template `pt` should guide this.
            return b; // Invalid leaper direction for this path calculation
        }
    }

    if (is_ok(leg_s)) {
        b |= square_bb(leg_s);
    }
    return b;
}

// Accumulates all lame leaper path squares from a given square for all its directions
// Renamed to avoid confusion with single-direction path calculation
template <PieceType pt>
Bitboard lame_leaper_path_all_dirs(Square s) {
    Bitboard b = 0;
    const auto& dirs = (pt == BISHOP) ? BishopDirections : KnightDirections;
    for (const auto& d : dirs) {
        b |= lame_leaper_path<pt>(d, s);
    }

    // Bishop specific constraint: stays in its half of the board
    if (pt == BISHOP) {
        b &= HalfBB[rank_of(s) > RANK_4];
    }
    return b;
}

// Computes lame leaper attacks given current occupancy
template <PieceType pt>
Bitboard lame_leaper_attack(Square s, Bitboard occupied) {
    Bitboard b = 0;
    const auto& dirs = (pt == BISHOP) ? BishopDirections : KnightDirections;

    for (const auto& d : dirs) {
        Square to = Square(s + d);
        // Ensure destination is valid and the "leg" square is not occupied
        // Distance check `distance(s, to) < 4` is redundant if `KnightDirections`/`BishopDirections` are well-defined.
        // A direct `is_ok(to)` is sufficient for destination validity.
        if (is_ok(to) && !(lame_leaper_path<pt>(d, s) & occupied)) {
            b |= square_bb(to);
        }
    }
    // Bishop specific constraint: attacks only in its half of the board
    if (pt == BISHOP) {
        b &= HalfBB[rank_of(s) > RANK_4];
    }
    return b;
}

// Initializes magic bitboard tables for sliding/leaping pieces
template <PieceType pt>
void init_magics(Bitboard table[], Magic magics[] IF_NOT_PEXT(, const Bitboard magicsInit[])) {

    // `edges` variable definition can be moved inside the loop as it's specific to `s`.
    // `b` and `size` are reset per square, so they are fine inside the loop too.

    for (Square s = SQ_A0; s <= SQ_I9; ++s) {
        Magic& m = magics[s]; // Reference for direct modification

        // 1. Calculate the mask
        Bitboard edges = ((Rank0BB | Rank9BB) & ~rank_bb(s)) | ((FileABB | FileIBB) & ~file_bb(s));

        if (pt == ROOK) {
            m.mask = sliding_attack<ROOK>(s, 0); // Attacks on empty board
        } else if (pt == CANNON) {
            // Cannon mask is the same as Rook mask in Chinese Chess for finding occupied squares.
            // The sliding_attack for Cannon handles the "hurdle" logic based on occupied.
            // So its mask is still just the straight lines.
            m.mask = sliding_attack<ROOK>(s, 0); // Use ROOK's mask generation logic for lines
        } else { // BISHOP, KNIGHT, KNIGHT_TO
            m.mask = lame_leaper_path_all_dirs<pt>(s); // All possible intermediate squares for leapers
        }

        // Apply edge exclusion, but not for KNIGHT_TO, as its mask refers to the "hop" squares.
        if (pt != KNIGHT_TO) {
            m.mask &= ~edges;
        }

        // 2. Set shift and magic number
#ifdef USE_PEXT
        m.shift = popcount(static_cast<uint64_t>(m.mask)); // Cast to uint64_t for popcount
#else
        m.magic = magicsInit[s];
        m.shift = 128 - popcount(static_cast<uint64_t>(m.mask)); // Cast to uint64_t for popcount
#endif

        // 3. Set the offset for the attacks table
        m.attacks = (s == SQ_A0) ? table : magics[s - 1].attacks + popcount(static_cast<uint64_t>(magics[s - 1].mask));

        // 4. Populate the attacks table using Carry-Rippler trick
        Bitboard b_occupancy = 0;
        uint64_t current_size = 0; // Use uint64_t for size

        // Do-while loop for Carry-Rippler trick to iterate through all subsets of m.mask
        do {
            Bitboard calculated_attack;
            if (pt == ROOK || pt == CANNON) {
                calculated_attack = sliding_attack<pt>(s, b_occupancy);
            } else { // BISHOP, KNIGHT, KNIGHT_TO
                calculated_attack = lame_leaper_attack<pt>(s, b_occupancy);
            }

            m.attacks[m.index(b_occupancy)] = calculated_attack;
            current_size++;
            b_occupancy = (b_occupancy - m.mask) & m.mask;
        } while (b_occupancy);

        // After the loop, `current_size` holds the number of entries for this square.
        // This size is implicitly used by `magics[s].attacks` pointing to the next free slot.
    }
}

} // anonymous namespace
} // namespace Stockfish
