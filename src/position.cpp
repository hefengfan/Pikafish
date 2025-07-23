/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "position.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <utility>

#include "bitboard.h"
#include "misc.h"
#include "movegen.h"
#include "tt.h"
#include "uci.h"

using std::string;

namespace Stockfish {

namespace Zobrist {

Key psq[PIECE_NB][SQUARE_NB];
Key side, noPawns;
}

namespace {

constexpr std::string_view PieceToChar(" RACPNBK racpnbk");

static constexpr Piece Pieces[] = {W_ROOK, W_ADVISOR, W_CANNON, W_PAWN, W_KNIGHT, W_BISHOP, W_KING,
                                   B_ROOK, B_ADVISOR, B_CANNON, B_PAWN, B_KNIGHT, B_BISHOP, B_KING};
}  // namespace

std::ostream& operator<<(std::ostream& os, const Position& pos) {
    os << "\n +---+---+---+---+---+---+---+---+---+\n";

    for (Rank r = RANK_9; r >= RANK_0; --r)
    {
        for (File f = FILE_A; f <= FILE_I; ++f)
            os << " | " << PieceToChar[pos.piece_on(make_square(f, r))];

        os << " | " << int(r) << "\n +---+---+---+---+---+---+---+---+---+\n";
    }

    os << "   a   b   c   d   e   f   g   h   i\n"
       << "\nFen: " << pos.fen() << "\nKey: " << std::hex << std::uppercase << std::setfill('0')
       << std::setw(16) << pos.key() << std::setfill(' ') << std::dec << "\nCheckers: ";

    for (Bitboard b = pos.checkers(); b;)
        os << UCIEngine::square(pop_lsb(b)) << " ";

    return os;
}

void Position::init() {
    PRNG rng(1070372);

    for (Piece pc : Pieces)
        for (Square s = SQ_A0; s <= SQ_I9; ++s)
            Zobrist::psq[pc][s] = rng.rand<Key>();

    Zobrist::side    = rng.rand<Key>();
    Zobrist::noPawns = rng.rand<Key>();
}

Position& Position::set(const string& fenStr, StateInfo* si) {
    unsigned char      token;
    size_t             idx;
    Square             sq = SQ_A9;
    std::istringstream ss(fenStr);

    std::memset(this, 0, sizeof(Position));
    midEncoding[WHITE] = midEncoding[BLACK] = Eval::NNUE::Features::HalfKAv2_hm::BalanceEncoding;

    std::memset(si, 0, sizeof(StateInfo));
    st = si;

    ss >> std::noskipws;

    // Piece placement
    while ((ss >> token) && !isspace(token))
    {
        if (isdigit(token))
            sq += (token - '0') * EAST;
        else if (token == '/')
            sq += 2 * SOUTH;
        else if ((idx = PieceToChar.find(token)) != string::npos)
        {
            put_piece(Piece(idx), sq);
            if (type_of(Piece(idx)) == KING)
                kingSquare[color_of(Piece(idx))] = sq;
            ++sq;
        }
    }

    // Active color
    ss >> token;
    sideToMove = (token == 'w' ? WHITE : BLACK);
    ss >> token;

    while ((ss >> token) && !isspace(token));
    while ((ss >> token) && !isspace(token));

    // Halfmove clock and fullmove number
    ss >> std::skipws >> st->rule60 >> gamePly;
    gamePly = std::max(2 * (gamePly - 1), 0) + (sideToMove == BLACK);

    set_state();
    assert(pos_is_ok());
    return *this;
}

void Position::set_check_info() const {
    update_blockers<WHITE>();
    update_blockers<BLACK>();

    Square ksq = king_square(~sideToMove);
    st->needSlowCheck = checkers() || (attacks_bb<ROOK>(king_square(sideToMove)) & pieces(~sideToMove, CANNON));

    st->checkSquares[PAWN]   = attacks_bb<PAWN_TO>(ksq, sideToMove);
    st->checkSquares[KNIGHT] = attacks_bb<KNIGHT_TO>(ksq, pieces());
    st->checkSquares[CANNON] = attacks_bb<CANNON>(ksq, pieces());
    st->checkSquares[ROOK]   = attacks_bb<ROOK>(ksq, pieces());
    st->checkSquares[KING] = st->checkSquares[ADVISOR] = st->checkSquares[BISHOP] = 0;

    Bitboard hollowCannons = st->checkSquares[ROOK] & pieces(sideToMove, CANNON);
    if (hollowCannons)
    {
        Bitboard hollowCannonDiscover = 0;
        while (hollowCannons)
            hollowCannonDiscover |= between_bb(ksq, pop_lsb(hollowCannons));
        for (PieceType pt = ROOK; pt < KING; ++pt)
            st->checkSquares[pt] |= hollowCannonDiscover;
    }
}

void Position::set_state() const {
    st->key = st->minorPieceKey = 0;
    st->nonPawnKey[WHITE] = st->nonPawnKey[BLACK] = 0;
    st->pawnKey = Zobrist::noPawns;
    st->majorMaterial[WHITE] = st->majorMaterial[BLACK] = VALUE_ZERO;
    st->checkersBB = checkers_to(~sideToMove, king_square(sideToMove));
    st->move = Move::none();

    set_check_info();

    for (Bitboard b = pieces(); b;)
    {
        Square s = pop_lsb(b);
        Piece pc = piece_on(s);
        PieceType pt = type_of(pc);
        st->key ^= Zobrist::psq[pc][s];

        if (pt == PAWN)
            st->pawnKey ^= Zobrist::psq[pc][s];
        else
        {
            st->nonPawnKey[color_of(pc)] ^= Zobrist::psq[pc][s];
            if (pt != KING && (pt & 1))
            {
                st->majorMaterial[color_of(pc)] += PieceValue[pc];
                if (pt != ROOK)
                    st->minorPieceKey ^= Zobrist::psq[pc][s];
            }
        }
    }

    if (sideToMove == BLACK)
        st->key ^= Zobrist::side;
}

string Position::fen() const {
    int emptyCnt;
    std::ostringstream ss;

    for (Rank r = RANK_9; r >= RANK_0; --r)
    {
        for (File f = FILE_A; f <= FILE_I; ++f)
        {
            for (emptyCnt = 0; f <= FILE_I && empty(make_square(f, r)); ++f)
                ++emptyCnt;

            if (emptyCnt)
                ss << emptyCnt;

            if (f <= FILE_I)
                ss << PieceToChar[piece_on(make_square(f, r))];
        }

        if (r > RANK_0)
            ss << '/';
    }

    ss << (sideToMove == WHITE ? " w " : " b ");
    ss << "- - " << st->rule60 << " " << 1 + (gamePly - (sideToMove == BLACK)) / 2;
    return ss.str();
}

template<Color c>
void Position::update_blockers() const {
    Square ksq = king_square(c);
    st->blockersForKing[c] = 0;
    st->pinners[~c] = 0;

    Bitboard snipers = ((attacks_bb<ROOK>(ksq) & (pieces(ROOK) | pieces(CANNON) | pieces(KING)))
                       | (attacks_bb<KNIGHT>(ksq) & pieces(KNIGHT)))
                      & pieces(~c);
    Bitboard occupancy = pieces() ^ (snipers & ~pieces(CANNON));

    while (snipers)
    {
        Square sniperSq = pop_lsb(snipers);
        bool isCannon = type_of(piece_on(sniperSq)) == CANNON;
        Bitboard b = between_bb(ksq, sniperSq) & (isCannon ? pieces() ^ sniperSq : occupancy);

        if (b && ((!isCannon && !more_than_one(b)) || (isCannon && popcount(b) == 2)))
        {
            st->blockersForKing[c] |= b;
            if (b & pieces(c))
                st->pinners[~c] |= sniperSq;
        }
    }
}

Bitboard Position::attackers_to(Square s, Bitboard occupied) const {
    return (attacks_bb<PAWN_TO>(s, WHITE) & pieces(WHITE, PAWN))
         | (attacks_bb<PAWN_TO>(s, BLACK) & pieces(BLACK, PAWN))
         | (attacks_bb<KNIGHT_TO>(s, occupied) & pieces(KNIGHT))
         | (attacks_bb<ROOK>(s, occupied) & pieces(ROOK))
         | (attacks_bb<CANNON>(s, occupied) & pieces(CANNON))
         | (attacks_bb<BISHOP>(s, occupied) & pieces(BISHOP))
         | (attacks_bb<ADVISOR>(s) & pieces(ADVISOR))
         | (attacks_bb<KING>(s) & pieces(KING));
}

Bitboard Position::checkers_to(Color c, Square s, Bitboard occupied) const {
    return ((attacks_bb<PAWN_TO>(s, c) & pieces(PAWN))
           | (attacks_bb<KNIGHT_TO>(s, occupied) & pieces(KNIGHT))
           | (attacks_bb<ROOK>(s, occupied) & pieces(KING, ROOK))
           | (attacks_bb<CANNON>(s, occupied) & pieces(CANNON)))
          & pieces(c);
}

bool Position::legal(Move m) const {
    assert(m.is_ok());
    Color us = sideToMove;
    Square from = m.from_sq();
    Square to = m.to_sq();
    Bitboard occupied = (pieces() ^ from) | to;

    assert(color_of(moved_piece(m)) == us);
    assert(piece_on(king_square(us)) == make_piece(us, KING));

    if (type_of(piece_on(from)) == KING)
        return !(checkers_to(~us, to, occupied));

    if (!st->needSlowCheck
        && (!(blockers_for_king(us) & from)
            || (((type_of(piece_on(from)) != CANNON) || !capture(m))
                && aligned(from, to, king_square(us)))))
        return true;

    return !(checkers_to(~us, king_square(us), occupied) & ~square_bb(to));
}

bool Position::pseudo_legal(const Move m) const {
    Color us = sideToMove;
    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece pc = moved_piece(m);

    if (pc == NO_PIECE || color_of(pc) != us)
        return false;
    if (pieces(us) & to)
        return false;

    if (type_of(pc) == PAWN)
        return bool(attacks_bb<PAWN>(from, us) & to);
    if (type_of(pc) == CANNON && !capture(m))
        return bool(attacks_bb<ROOK>(from, pieces()) & to);
    
    return bool(attacks_bb(type_of(pc), from, pieces()) & to);
}

bool Position::gives_check(Move m) const {
    assert(m.is_ok());
    assert(color_of(moved_piece(m)) == sideToMove);

    Square from = m.from_sq();
    Square to = m.to_sq();
    Square ksq = king_square(~sideToMove);
    PieceType pt = type_of(moved_piece(m));

    if (pt == CANNON && aligned(from, to, ksq))
    {
        if (attacks_bb<CANNON>(to, (pieces() ^ from) | to) & ksq)
            return true;
    }
    else if (check_squares(pt) & to)
        return true;

    if (attacks_bb<ROOK>(ksq) & pieces(sideToMove, CANNON))
        return bool(checkers_to(sideToMove, ksq, (pieces() ^ from) | to) & ~square_bb(from));
    if ((blockers_for_king(~sideToMove) & from) && !aligned(from, to, ksq))
        return true;

    return false;
}

DirtyPiece Position::do_move(Move m, StateInfo& newSt, bool givesCheck, const TranspositionTable* tt) {
    assert(m.is_ok());
    assert(&newSt != st);

    ++filter[st->key];
    Key k = st->key ^ Zobrist::side;

    std::memcpy(&newSt, st, offsetof(StateInfo, key));
    newSt.previous = st;
    st = &newSt;
    st->move = m;

    ++gamePly;
    if (!givesCheck || ++st->check10[sideToMove] <= 10)
    {
        if (st->check10[~sideToMove] > 10 && st->previous->checkersBB)
            ++st->check10[~sideToMove];
        else
            ++st->rule60;
    }
    ++st->pliesFromNull;

    Color us = sideToMove;
    Color them = ~us;
    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece pc = piece_on(from);
    Piece captured = piece_on(to);

    DirtyPiece dp;
    dp.pc = pc;
    dp.from = from;
    dp.to = to;

    assert(color_of(pc) == us);
    assert(captured == NO_PIECE || color_of(captured) == them);
    assert(type_of(captured) != KING);

    if (pc == make_piece(us, KING))
    {
        dp.requires_refresh[us] = true;
        bool mirror_before = Eval::NNUE::FeatureSet::KingBuckets[king_square(them)][from][0].second;
        bool mirror_after = Eval::NNUE::FeatureSet::KingBuckets[king_square(them)][to][0].second;
        dp.requires_refresh[them] = (mirror_before != mirror_after);
    }
    else
        dp.requires_refresh[us] = dp.requires_refresh[them] = false;

    bool mid_mirror_before[2] = {Eval::NNUE::FeatureSet::requires_mid_mirror(*this, us),
                                 Eval::NNUE::FeatureSet::requires_mid_mirror(*this, them)};

    if (captured)
    {
        Square capsq = to;

        if (type_of(captured) == PAWN)
            st->pawnKey ^= Zobrist::psq[captured][capsq];
        else
        {
            st->nonPawnKey[them] ^= Zobrist::psq[captured][capsq];
            if (type_of(captured) & 1)
            {
                st->majorMaterial[them] -= PieceValue[captured];
                if (type_of(captured) != ROOK)
                    st->minorPieceKey ^= Zobrist::psq[captured][capsq];
            }
        }

        dp.remove_pc = captured;
        dp.remove_sq = capsq;

        auto attack_bucket_before = Eval::NNUE::FeatureSet::make_attack_bucket(*this, them);
        remove_piece(capsq);
        auto attack_bucket_after = Eval::NNUE::FeatureSet::make_attack_bucket(*this, them);

        if (attack_bucket_before != attack_bucket_after)
            dp.requires_refresh[them] = true;

        k ^= Zobrist::psq[captured][capsq];
        st->check10[WHITE] = st->check10[BLACK] = st->rule60 = 0;
    }
    else
        dp.remove_sq = SQ_NONE;

    k ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
    if (type_of(pc) == PAWN)
        st->pawnKey ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
    else
    {
        st->nonPawnKey[us] ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
        if (type_of(pc) == KNIGHT || type_of(pc) == CANNON)
            st->minorPieceKey ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
    }

    move_piece(from, to);

    dp.requires_refresh[us] |= (mid_mirror_before[0] != Eval::NNUE::FeatureSet::requires_mid_mirror(*this, us));
    dp.requires_refresh[them] |= (mid_mirror_before[1] != Eval::NNUE::FeatureSet::requires_mid_mirror(*this, them));

    st->key = k;
    if (tt)
        prefetch(tt->first_entry(key()));

    st->capturedPiece = captured;
    st->checkersBB = givesCheck ? checkers_to(us, king_square(them)) : Bitboard(0);
    assert(givesCheck == bool(st->checkersBB));

    sideToMove = ~sideToMove;
    set_check_info();
    assert(pos_is_ok());

    assert(dp.pc != NO_PIECE);
    assert(!bool(captured) ^ (dp.remove_sq != SQ_NONE));
    assert(dp.from != SQ_NONE && dp.to != SQ_NONE);
    return dp;
}

void Position::undo_move(Move m) {
    assert(m.is_ok());
    sideToMove = ~sideToMove;

    Square from = m.from_sq();
    Square to = m.to_sq();

    assert(empty(from));
    assert(type_of(st->capturedPiece) != KING);

    move_piece(to, from);

    if (st->capturedPiece)
        put_piece(st->capturedPiece, to);

    st = st->previous;
    --gamePly;
    --filter[st->key];
    assert(pos_is_ok());
}

void Position::do_null_move(StateInfo& newSt, const TranspositionTable& tt) {
    assert(!checkers());
    assert(&newSt != st);

    ++filter[st->key];
    std::memcpy(&newSt, st, sizeof(StateInfo));

    newSt.previous = st;
    st = &newSt;
    st->key ^= Zobrist::side;
    prefetch(tt.first_entry(key()));
    st->pliesFromNull = 0;
    sideToMove = ~sideToMove;
    set_check_info();
    assert(pos_is_ok());
}

void Position::undo_null_move() {
    assert(!checkers());
    st = st->previous;
    sideToMove = ~sideToMove;
    --filter[st->key];
}

bool Position::see_ge(Move m, int threshold) const {
    assert(m.is_ok());
    Square from = m.from_sq(), to = m.to_sq();

    int swap = PieceValue[piece_on(to)] - threshold;
    if (swap < 0)
        return false;

    swap = PieceValue[piece_on(from)] - swap;
    if (swap <= 0)
        return true;

    assert(color_of(piece_on(from)) == sideToMove);
    Bitboard occupied = pieces() ^ from ^ to;
    Color stm = sideToMove;
    Bitboard attackers = attackers_to(to, occupied);

    if (attackers & pieces(stm, KING))
        attackers |= attacks_bb<ROOK>(to, occupied & ~pieces(ROOK)) & pieces(~stm, KING);
    if (attackers & pieces(~stm, KING))
        attackers |= attacks_bb<ROOK>(to, occupied & ~pieces(ROOK)) & pieces(stm, KING);

    Bitboard nonCannons = attackers & ~pieces(CANNON);
    Bitboard cannons = attackers & pieces(CANNON);
    Bitboard stmAttackers, bb;
    int res = 1;

    while (true)
    {
        stm = ~stm;
        attackers &= occupied;

        if (!(stmAttackers = attackers & pieces(stm)))
            break;

        if (pinners(~stm) & occupied)
        {
            stmAttackers &= ~blockers_for_king(stm);
            if (!stmAttackers)
                break;
        }

        res ^= 1;

        if ((bb = stmAttackers & pieces(PAWN)))
        {
            if ((swap = PawnValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);
            nonCannons |= attacks_bb<ROOK>(to, occupied) & pieces(ROOK);
            cannons = attacks_bb<CANNON>(to, occupied) & pieces(CANNON);
            attackers = nonCannons | cannons;
        }
        else if ((bb = stmAttackers & pieces(BISHOP)))
        {
            if ((swap = BishopValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);
        }
        else if ((bb = stmAttackers & pieces(ADVISOR)))
        {
            if ((swap = AdvisorValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);
            nonCannons |= attacks_bb<KNIGHT_TO>(to, occupied) & pieces(KNIGHT);
            attackers = nonCannons | cannons;
        }
        else if ((bb = stmAttackers & pieces(CANNON)))
        {
            if ((swap = CannonValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);
            cannons = attacks_bb<CANNON>(to, occupied) & pieces(CANNON);
            attackers = nonCannons | cannons;
        }
        else if ((bb = stmAttackers & pieces(KNIGHT)))
        {
            if ((swap = KnightValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);
        }
        else if ((bb = stmAttackers & pieces(ROOK)))
        {
            swap = RookValue - swap;
            occupied ^= least_significant_square_bb(bb);
            nonCannons |= attacks_bb<ROOK>(to, occupied) & pieces(ROOK);
            cannons = attacks_bb<CANNON>(to, occupied) & pieces(CANNON);
            attackers = nonCannons | cannons;
        }
        else
            return (attackers & ~pieces(stm)) ? res ^ 1 : res;
    }

    return bool(res);
}

std::pair<Piece, int> Position::do_move(Move m) {
    assert(capture(m));

    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece captured = piece_on(to);
    int id = idBoard[to];

    idBoard[to] = idBoard[from];
    idBoard[from] = 0;

    remove_piece(to);
    move_piece(from, to);

    sideToMove = ~sideToMove;
    return {captured, id};
}

void Position::undo_move(Move m, Piece captured, int id) {
    sideToMove = ~sideToMove;

    Square from = m.from_sq();
    Square to = m.to_sq();

    idBoard[from] = idBoard[to];
    idBoard[to] = id;

    move_piece(to, from);
    if (captured)
        put_piece(captured, to);
}

bool Position::chase_legal(Move m) const {
    assert(m.is_ok());
    Color us = sideToMove;
    Square from = m.from_sq();
    Square to = m.to_sq();
    Bitboard occupied = (pieces() ^ from) | to;

    assert(color_of(moved_piece(m)) == us);
    assert(piece_on(king_square(us)) == make_piece(us, KING));

    if (type_of(piece_on(from)) == KING)
        return !(checkers_to(~us, to, occupied));

    return !(checkers_to(~us, king_square(us), occupied) & ~square_bb(to));
}

uint16_t Position::chased(Color c) {
    uint16_t chase = 0;
    std::swap(c, sideToMove);

    Bitboard attackers = pieces(sideToMove) ^ pieces(sideToMove, KING, PAWN);
    while (attackers)
    {
        Square from = pop_lsb(attackers);
        PieceType attackerType = type_of(piece_on(from));
        Bitboard attacks = attacks_bb(attackerType, from, pieces());

        if (blockers_for_king(sideToMove) & from)
            attacks &= pinners(~sideToMove) & ~pieces(KING);
        else
            attacks &= (pieces(~sideToMove) ^ pieces(~sideToMove, KING, PAWN))
                     | (pieces(~sideToMove, PAWN) & HalfBB[sideToMove]);

        while (attacks)
        {
            Square to = pop_lsb(attacks);
            Move m = Move(from, to);

            if (chase_legal(m))
            {
                if ((attackerType == KNIGHT || attackerType == CANNON)
                    && type_of(piece_on(to)) == ROOK)
                    chase |= (1 << idBoard[to]);
                if ((attackerType == ADVISOR || attackerType == BISHOP)
                    && type_of(piece_on(to)) & 1)
                    chase |= (1 << idBoard[to]);
                else
                {
                    bool trueChase = true;
                    const auto& [captured, id] = do_move(m);
                    Bitboard recaptures = attackers_to(to) & pieces(sideToMove);
                    while (recaptures)
                    {
                        Square s = pop_lsb(recaptures);
                        if (chase_legal(Move(s, to)))
                        {
                            trueChase = false;
                            break;
                        }
                    }
                    undo_move(m, captured, id);

                    if (trueChase)
                    {
                        if (attackerType == type_of(piece_on(to)))
                        {
                            sideToMove = ~sideToMove;
                            if ((attackerType == KNIGHT && ((between_bb(from, to) ^ to) & pieces()))
                                || !chase_legal(Move(to, from)))
                                chase |= (1 << idBoard[to]);
                            sideToMove = ~sideToMove;
                        }
                        else
                            chase |= (1 << idBoard[to]);
                    }
                }
            }
        }
    }

    std::swap(c, sideToMove);
    return chase;
}

Value Position::detect_chases(int d, int ply) {
    int whiteId = 0;
    int blackId = 0;
    for (Square s = SQ_A0; s <= SQ_I9; ++s)
        if (board[s] != NO_PIECE)
            idBoard[s] = color_of(board[s]) == WHITE ? whiteId++ : blackId++;

    Color us = sideToMove, them = ~us;
    uint16_t chase[COLOR_NB] = {0xFFFF, 0xFFFF};

    for (int i = 0; i < d; ++i)
    {
        if (st->checkersBB)
            return VALUE_DRAW;
        else if (!chase[~sideToMove])
        {
            if (!chase[sideToMove])
                break;
            undo_move(st->move, st->capturedPiece);
            st = st->previous;
        }
        else
        {
            uint16_t after = chased(~sideToMove);
            undo_move(st->move, st->capturedPiece);
            st = st->previous;
            chase[sideToMove] &= after & ~chased(sideToMove);
        }
    }

    return bool(chase[us]) ^ bool(chase[them]) ? chase[us] ? mated_in(ply) : mate_in(ply)
                                               : VALUE_DRAW;
}

bool Position::rule_judge(Value& result, int ply) {
    int end = std::min(st->rule60 + std::max(0, st->check10[WHITE] - 10)
                         + std::max(0, st->check10[BLACK] - 10),
                       st->pliesFromNull);

    if (end >= 4 && filter[st->key] >= 1)
    {
        int cnt = 0;
        StateInfo* stp = st->previous->previous;
        bool checkThem = st->checkersBB && stp->checkersBB;
        bool checkUs = st->previous->checkersBB && stp->previous->checkersBB;

        for (int i = 4; i <= end; i += 2)
        {
            stp = stp->previous->previous;
            checkThem &= bool(stp->checkersBB);

            if (stp->key == st->key && (++cnt == 2 || ply > i))
            {
                if (!checkThem && !checkUs)
                {
                    Position rollback;
                    memcpy((void*) &rollback, (const void*) this, offsetof(Position, filter));
                    result = rollback.detect_chases(i, ply);
                }
                else
                    result = !checkUs ? mate_in(ply) : !checkThem ? mated_in(ply) : VALUE_DRAW;

                if (result == VALUE_DRAW || cnt == 2)
                    return true;

                if (filter[st->key] <= 1)
                {
                    if (st->rule60 < 120 && st->previous->key == stp->previous->key)
                    {
                        StateInfo* prev = st->previous;
                        while ((prev = prev->previous) != stp)
                            if (filter[prev->key] > 1)
                                break;
                        if (prev == stp)
                            return true;
                    }
                    break;
                }
            }

            if (i + 1 <= end)
                checkUs &= bool(stp->previous->checkersBB);
        }
    }

    if (st->rule60 >= 120)
    {
        result = MoveList<LEGAL>(*this).size() ? VALUE_DRAW : mated_in(ply);
        return true;
    }

    if (count<PAWN>() == 0)
    {
        enum DrawLevel : int { NO_DRAW, DIRECT_DRAW, MATE_DRAW };
        int level = [&]() {
            if (!major_material())
                return DIRECT_DRAW;

            if (major_material() == CannonValue)
            {
                Color cannonSide = major_material(WHITE) == CannonValue ? WHITE : BLACK;
                if (count<ADVISOR>(cannonSide) == 0)
                {
                    if (count<ADVISOR>(~cannonSide) == 0)
                        return DIRECT_DRAW;
                    if (count<ADVISOR>(~cannonSide) == 1)
                        return count<BISHOP>(cannonSide) == 0 ? DIRECT_DRAW : MATE_DRAW;
                    if (count<BISHOP>(cannonSide) == 0)
                        return MATE_DRAW;
                }
            }

            if (major_material(WHITE) == CannonValue && major_material(BLACK) == CannonValue
                && count<ADVISOR>() == 0)
                return count<BISHOP>() == 0 ? DIRECT_DRAW : MATE_DRAW;

            return NO_DRAW;
        }();

        if (level != NO_DRAW)
        {
            if (level == MATE_DRAW)
            {
                MoveList<LEGAL> moves(*this);
                if (moves.size() == 0)
                {
                    result = mated_in(ply);
                    return true;
                }
                for (const auto& move : moves)
                {
                    StateInfo tempSt;
                    do_move(move, tempSt);
                    bool mate = MoveList<LEGAL>(*this).size() == 0;
                    undo_move(move);
                    if (mate)
                        return false;
                }
            }
            result = VALUE_DRAW;
            return true;
        }
    }

    return false;
}

void Position::flip() {
    string f, token;
    std::stringstream ss(fen());

    for (Rank r = RANK_9; r >= RANK_0; --r)
    {
        std::getline(ss, token, r > RANK_0 ? '/' : ' ');
        f.insert(0, token + (f.empty() ? " " : "/"));
    }

    ss >> token;
    f += (token == "w" ? "B " : "W ");

    ss >> token;
    f += token + " ";

    std::transform(f.begin(), f.end(), f.begin(),
                   [](char c) { return char(islower(c) ? toupper(c) : tolower(c)); });

    ss >> token;
    f += token;

    std::getline(ss, token);
    f += token;

    set(f, st);
    assert(pos_is_ok());
}

bool Position::pos_is_ok() const {
    constexpr bool Fast = true;

    if ((sideToMove != WHITE && sideToMove != BLACK) || piece_on(king_square(WHITE)) != W_KING
        || piece_on(king_square(BLACK)) != B_KING)
        assert(0 && "pos_is_ok: Default");

    if (Fast)
        return true;

    if (pieceCount[W_KING] != 1 || pieceCount[B_KING] != 1
        || checkers_to(sideToMove, king_square(~sideToMove)))
        assert(0 && "pos_is_ok: Kings");

    if ((pieces(WHITE, PAWN) & ~PawnBB[WHITE]) || (pieces(BLACK, PAWN) & ~PawnBB[BLACK])
        || pieceCount[W_PAWN] > 5 || pieceCount[B_PAWN] > 5)
        assert(0 && "pos_is_ok: Pawns");

    if ((pieces(WHITE) & pieces(BLACK)) || (pieces(WHITE) | pieces(BLACK)) != pieces()
        || popcount(pieces(WHITE)) > 16 || popcount(pieces(BLACK)) > 16)
        assert(0 && "pos_is_ok: Bitboards");

    for (PieceType p1 = PAWN; p1 <= KING; ++p1)
        for (PieceType p2 = PAWN; p2 <= KING; ++p2)
            if (p1 != p2 && (pieces(p1) & pieces(p2)))
                assert(0 && "pos_is_ok: Bitboards");

    for (Piece pc : Pieces)
        if (pieceCount[pc] != popcount(pieces(color_of(pc), type_of(pc)))
            || pieceCount[pc] != std::count(board, board + SQUARE_NB, pc))
            assert(0 && "pos_is_ok: Pieces");

    return true;
}

}  // namespace Stockfish
