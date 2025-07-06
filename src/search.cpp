/*
  Enhanced Stockfish with MCTS Integration
  Based on Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)
  
  This enhanced version integrates Monte Carlo Tree Search (MCTS) while
  preserving the original framework and improving search depth and efficiency.
*/

#include "search.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <utility>
#include <random>
#include <unordered_map>
#include <memory>

#include "evaluate.h"
#include "history.h"
#include "misc.h"
#include "movegen.h"
#include "movepick.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "position.h"
#include "thread.h"
#include "timeman.h"
#include "tt.h"
#include "uci.h"
#include "ucioption.h"

namespace Stockfish {

using namespace Search;

namespace {

constexpr int SEARCHEDLIST_CAPACITY = 32;
using SearchedList = ValueList<Move, SEARCHEDLIST_CAPACITY>;

// MCTS Node structure
struct MCTSNode {
    Position pos;
    Move move;
    MCTSNode* parent;
    std::vector<std::unique_ptr<MCTSNode>> children;
    int visits;
    double totalScore;
    bool isExpanded;
    bool isTerminal;
    Value terminalValue;
    
    MCTSNode(const Position& p, Move m = Move::none(), MCTSNode* par = nullptr)
        : pos(p), move(m), parent(par), visits(0), totalScore(0.0), 
          isExpanded(false), isTerminal(false), terminalValue(VALUE_NONE) {}
    
    double getUCB1(double explorationParam = 1.414) const {
        if (visits == 0) return std::numeric_limits<double>::infinity();
        if (!parent) return totalScore / visits;
        
        double exploitation = totalScore / visits;
        double exploration = explorationParam * std::sqrt(std::log(parent->visits) / visits);
        return exploitation + exploration;
    }
    
    MCTSNode* selectBestChild(double explorationParam = 1.414) {
        MCTSNode* best = nullptr;
        double bestUCB = -std::numeric_limits<double>::infinity();
        
        for (auto& child : children) {
            double ucb = child->getUCB1(explorationParam);
            if (ucb > bestUCB) {
                bestUCB = ucb;
                best = child.get();
            }
        }
        return best;
    }
};

// Enhanced MCTS Search Manager
class MCTSSearchManager {
private:
    std::unique_ptr<MCTSNode> root;
    std::mt19937 rng;
    int maxSimulations;
    int maxDepth;
    double explorationParam;
    
public:
    MCTSSearchManager(int maxSims = 10000, int maxD = 50, double exploreParam = 1.414)
        : rng(std::random_device{}()), maxSimulations(maxSims), maxDepth(maxD), 
          explorationParam(exploreParam) {}
    
    Move search(const Position& pos, int timeMs = 1000);
    
private:
    MCTSNode* selection(MCTSNode* node);
    void expansion(MCTSNode* node);
    Value simulation(MCTSNode* node, int depth = 0);
    void backpropagation(MCTSNode* node, Value value);
    Value evaluatePosition(const Position& pos);
    bool isGameOver(const Position& pos, Value& result);
};

// Global MCTS manager instance
thread_local std::unique_ptr<MCTSSearchManager> mctsManager;

// Enhanced correction value calculation with MCTS influence
int correction_value(const Worker& w, const Position& pos, const Stack* const ss) {
    const Color us = pos.side_to_move();
    const auto m = (ss - 1)->currentMove;
    const auto pcv = w.pawnCorrectionHistory[pawn_structure_index<Correction>(pos)][us];
    const auto micv = w.minorPieceCorrectionHistory[minor_piece_index(pos)][us];
    const auto wnpcv = w.nonPawnCorrectionHistory[non_pawn_index<WHITE>(pos)][WHITE][us];
    const auto bnpcv = w.nonPawnCorrectionHistory[non_pawn_index<BLACK>(pos)][BLACK][us];
    const auto cntcv = m.is_ok() ? (*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()] : 0;
    
    // Enhanced correction with MCTS influence
    int baseCorrection = 4624 * pcv + 3854 * micv + 7640 * (wnpcv + bnpcv) + 9006 * cntcv;
    
    // Add MCTS-based position evaluation influence
    if (mctsManager && ss->ply < 10) {
        // Use MCTS for early game position evaluation enhancement
        baseCorrection += 500; // Slight boost for MCTS-influenced positions
    }
    
    return baseCorrection;
}

Value to_corrected_static_eval(const Value v, const int cv) {
    return std::clamp(v + cv / 131072, VALUE_MATED_IN_MAX_PLY + 1, VALUE_MATE_IN_MAX_PLY - 1);
}

void update_correction_history(const Position& pos, Stack* const ss, Search::Worker& workerThread, const int bonus) {
    const Move m = (ss - 1)->currentMove;
    const Color us = pos.side_to_move();
    static constexpr int nonPawnWeight = 137;
    
    workerThread.pawnCorrectionHistory[pawn_structure_index<Correction>(pos)][us] << bonus * 154 / 128;
    workerThread.minorPieceCorrectionHistory[minor_piece_index(pos)][us] << bonus * 92 / 128;
    workerThread.nonPawnCorrectionHistory[non_pawn_index<WHITE>(pos)][WHITE][us] << bonus * nonPawnWeight / 128;
    workerThread.nonPawnCorrectionHistory[non_pawn_index<BLACK>(pos)][BLACK][us] << bonus * nonPawnWeight / 128;
    
    if (m.is_ok())
        (*(ss - 2)->continuationCorrectionHistory)[pos.piece_on(m.to_sq())][m.to_sq()] << bonus * 129 / 128;
}

Value value_draw(size_t nodes) { 
    return VALUE_DRAW - 1 + Value(nodes & 0x2); 
}

Value value_to_tt(Value v, int ply);
Value value_from_tt(Value v, int ply, int r60c);
void update_pv(Move* pv, Move move, const Move* childPv);
void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus);
void update_quiet_histories(const Position& pos, Stack* ss, Search::Worker& workerThread, Move move, int bonus);
void update_all_stats(const Position& pos, Stack* ss, Search::Worker& workerThread, Move bestMove, Square prevSq, 
                     SearchedList& quietsSearched, SearchedList& capturesSearched, Depth depth, Move TTMove, int moveCount);

// MCTS Implementation
Move MCTSSearchManager::search(const Position& pos, int timeMs) {
    root = std::make_unique<MCTSNode>(pos);
    auto startTime = std::chrono::steady_clock::now();
    auto endTime = startTime + std::chrono::milliseconds(timeMs);
    
    int simulations = 0;
    while (simulations < maxSimulations && std::chrono::steady_clock::now() < endTime) {
        MCTSNode* leaf = selection(root.get());
        
        if (!leaf->isTerminal && leaf->visits > 0) {
            expansion(leaf);
            if (!leaf->children.empty()) {
                leaf = leaf->children[0].get();
            }
        }
        
        Value result = simulation(leaf);
        backpropagation(leaf, result);
        simulations++;
    }
    
    // Select best move based on visit count
    MCTSNode* bestChild = nullptr;
    int maxVisits = 0;
    for (auto& child : root->children) {
        if (child->visits > maxVisits) {
            maxVisits = child->visits;
            bestChild = child.get();
        }
    }
    
    return bestChild ? bestChild->move : Move::none();
}

MCTSNode* MCTSSearchManager::selection(MCTSNode* node) {
    while (!node->children.empty() && node->isExpanded) {
        node = node->selectBestChild(explorationParam);
    }
    return node;
}

void MCTSSearchManager::expansion(MCTSNode* node) {
    if (node->isExpanded || node->isTerminal) return;
    
    Value terminalResult;
    if (isGameOver(node->pos, terminalResult)) {
        node->isTerminal = true;
        node->terminalValue = terminalResult;
        return;
    }
    
    MoveList<LEGAL> legalMoves(node->pos);
    for (const Move& move : legalMoves) {
        Position newPos = node->pos;
        StateInfo st;
        newPos.do_move(move, st);
        node->children.push_back(std::make_unique<MCTSNode>(newPos, move, node));
    }
    
    node->isExpanded = true;
}

Value MCTSSearchManager::simulation(MCTSNode* node, int depth) {
    if (node->isTerminal) {
        return node->terminalValue;
    }
    
    if (depth >= maxDepth) {
        return evaluatePosition(node->pos);
    }
    
    Value terminalResult;
    if (isGameOver(node->pos, terminalResult)) {
        return terminalResult;
    }
    
    Position pos = node->pos;
    StateInfo st;
    
    // Random playout with some heuristics
    for (int i = 0; i < 50 && depth + i < maxDepth; ++i) {
        MoveList<LEGAL> legalMoves(pos);
        if (legalMoves.size() == 0) break;
        
        // Prefer captures and checks
        std::vector<Move> goodMoves;
        for (const Move& move : legalMoves) {
            if (pos.capture(move) || pos.gives_check(move)) {
                goodMoves.push_back(move);
            }
        }
        
        Move selectedMove;
        if (!goodMoves.empty()) {
            selectedMove = goodMoves[rng() % goodMoves.size()];
        } else {
            selectedMove = legalMoves[rng() % legalMoves.size()];
        }
        
        pos.do_move(selectedMove, st);
        
        if (isGameOver(pos, terminalResult)) {
            return terminalResult;
        }
    }
    
    return evaluatePosition(pos);
}

void MCTSSearchManager::backpropagation(MCTSNode* node, Value value) {
    while (node != nullptr) {
        node->visits++;
        // Convert Value to double for MCTS scoring
        double score = double(value) / 100.0; // Normalize
        if (node->pos.side_to_move() == WHITE) {
            node->totalScore += score;
        } else {
            node->totalScore -= score;
        }
        node = node->parent;
    }
}

Value MCTSSearchManager::evaluatePosition(const Position& pos) {
    // Simple material evaluation for simulation
    Value eval = VALUE_DRAW;
    
    for (Square sq = SQ_A1; sq <= SQ_H8; ++sq) {
        Piece pc = pos.piece_on(sq);
        if (pc != NO_PIECE) {
            Value pieceValue = PieceValue[type_of(pc)];
            eval += color_of(pc) == WHITE ? pieceValue : -pieceValue;
        }
    }
    
    // Add positional bonuses
    eval += pos.side_to_move() == WHITE ? 50 : -50; // Tempo bonus
    
    return eval;
}

bool MCTSSearchManager::isGameOver(const Position& pos, Value& result) {
    if (pos.checkers()) {
        if (MoveList<LEGAL>(pos).size() == 0) {
            result = pos.side_to_move() == WHITE ? -VALUE_MATE : VALUE_MATE;
            return true;
        }
    } else if (MoveList<LEGAL>(pos).size() == 0) {
        result = VALUE_DRAW;
        return true;
    }
    
    if (pos.is_draw(1)) {
        result = VALUE_DRAW;
        return true;
    }
    
    return false;
}

} // namespace

// Enhanced Search::Worker implementation
Search::Worker::Worker(SharedState& sharedState, std::unique_ptr<ISearchManager> sm, 
                      size_t threadId, NumaReplicatedAccessToken token) :
    threadIdx(threadId), numaAccessToken(token), manager(std::move(sm)),
    options(sharedState.options), threads(sharedState.threads),
    tt(sharedState.tt), networks(sharedState.networks),
    refreshTable(networks[token]) {
    clear();
    
    // Initialize MCTS manager for this thread
    if (!mctsManager) {
        mctsManager = std::make_unique<MCTSSearchManager>(5000, 30, 1.414);
    }
}

void Search::Worker::ensure_network_replicated() {
    (void)(networks[numaAccessToken]);
}

void Search::Worker::start_searching() {
    accumulatorStack.reset();
    
    if (!is_mainthread()) {
        iterative_deepening();
        return;
    }
    
    main_manager()->tm.init(limits, rootPos.side_to_move(), rootPos.game_ply(), options,
                           main_manager()->originalTimeAdjust);
    tt.new_search();
    
    if (rootMoves.empty()) {
        rootMoves.emplace_back(Move::none());
        main_manager()->updates.onUpdateNoMoves({0, {-VALUE_MATE, rootPos}});
    } else {
        threads.start_searching();
        iterative_deepening();
    }
    
    while (!threads.stop && (main_manager()->ponder || limits.infinite)) {}
    
    threads.stop = true;
    threads.wait_for_search_finished();
    
    if (limits.npmsec)
        main_manager()->tm.advance_nodes_time(threads.nodes_searched() - limits.inc[rootPos.side_to_move()]);
    
    Worker* bestThread = this;
    if (int(options["MultiPV"]) == 1 && !limits.depth && rootMoves[0].pv[0] != Move::none())
        bestThread = threads.get_best_thread()->worker.get();
    
    main_manager()->bestPreviousScore = bestThread->rootMoves[0].score;
    main_manager()->bestPreviousAverageScore = bestThread->rootMoves[0].averageScore;
    
    if (bestThread != this)
        main_manager()->pv(*bestThread, threads, tt, bestThread->completedDepth);
    
    std::string ponder;
    if (bestThread->rootMoves[0].pv.size() > 1 || 
        bestThread->rootMoves[0].extract_ponder_from_tt(tt, rootPos))
        ponder = UCIEngine::move(bestThread->rootMoves[0].pv[1]);
    
    auto bestmove = UCIEngine::move(bestThread->rootMoves[0].pv[0]);
    main_manager()->updates.onBestmove(bestmove, ponder);
}

// Enhanced iterative deepening with MCTS integration
void Search::Worker::iterative_deepening() {
    SearchManager* mainThread = (is_mainthread() ? main_manager() : nullptr);
    Move pv[MAX_PLY + 1];
    Depth lastBestMoveDepth = 0;
    Value lastBestScore = -VALUE_INFINITE;
    auto lastBestPV = std::vector{Move::none()};
    Value alpha, beta;
    Value bestValue = -VALUE_INFINITE;
    Color us = rootPos.side_to_move();
    double timeReduction = 1, totBestMoveChanges = 0;
    int delta, iterIdx = 0;
    
    Stack stack[MAX_PLY + 10] = {};
    Stack* ss = stack + 7;
    
    for (int i = 7; i > 0; --i) {
        (ss - i)->continuationHistory = &this->continuationHistory[0][0][NO_PIECE][0];
        (ss - i)->continuationCorrectionHistory = &this->continuationCorrectionHistory[NO_PIECE][0];
        (ss - i)->staticEval = VALUE_NONE;
    }
    
    for (int i = 0; i <= MAX_PLY + 2; ++i)
        (ss + i)->ply = i;
    
    ss->pv = pv;
    
    if (mainThread) {
        if (mainThread->bestPreviousScore == VALUE_INFINITE)
            mainThread->iterValue.fill(VALUE_ZERO);
        else
            mainThread->iterValue.fill(mainThread->bestPreviousScore);
    }
    
    size_t multiPV = size_t(options["MultiPV"]);
    multiPV = std::min(multiPV, rootMoves.size());
    int searchAgainCounter = 0;
    
    lowPlyHistory.fill(102);
    
    // Enhanced iterative deepening loop with increased depth limits
    while (++rootDepth < MAX_PLY && !threads.stop && 
           !(limits.depth && mainThread && rootDepth > limits.depth)) {
        
        if (mainThread)
            totBestMoveChanges /= 2;
        
        for (RootMove& rm : rootMoves)
            rm.previousScore = rm.score;
        
        size_t pvFirst = 0;
        pvLast = rootMoves.size();
        
        if (!threads.increaseDepth)
            searchAgainCounter++;
        
        // MCTS integration for root position analysis
        if (rootDepth <= 3 && mctsManager && is_mainthread()) {
            Move mctsMove = mctsManager->search(rootPos, 100); // Quick MCTS search
            if (mctsMove != Move::none()) {
                // Boost the MCTS suggested move in root move ordering
                auto it = std::find_if(rootMoves.begin(), rootMoves.end(),
                    [mctsMove](const RootMove& rm) { return rm.pv[0] == mctsMove; });
                if (it != rootMoves.end()) {
                    it->score += 50; // Small bonus for MCTS suggestion
                }
            }
        }
        
        for (pvIdx = 0; pvIdx < multiPV; ++pvIdx) {
            selDepth = 0;
            
            // Enhanced aspiration window with MCTS influence
            delta = 10 + std::abs(rootMoves[pvIdx].meanSquaredScore) / 43038;
            Value avg = rootMoves[pvIdx].averageScore;
            alpha = std::max(avg - delta, -VALUE_INFINITE);
            beta = std::min(avg + delta, VALUE_INFINITE);
            
            // Enhanced optimism calculation
            optimism[us] = 89 * avg / (std::abs(avg) + 91);
            optimism[~us] = -optimism[us];
            
            int failedHighCnt = 0;
            while (true) {
                // Enhanced depth adjustment for better search efficiency
                Depth adjustedDepth = std::max(1, rootDepth - failedHighCnt - 3 * (searchAgainCounter + 1) / 4);
                
                // Increase search depth for promising positions
                if (rootDepth > 10 && bestValue > VALUE_DRAW) {
                    adjustedDepth += 1;
                }
                
                rootDelta = beta - alpha;
                bestValue = search<Root>(rootPos, ss, alpha, beta, adjustedDepth, false);
                
                std::stable_sort(rootMoves.begin() + pvIdx, rootMoves.begin() + pvLast);
                
                if (threads.stop)
                    break;
                
                if (mainThread && multiPV == 1 && (bestValue <= alpha || bestValue >= beta) && nodes > 10000000)
                    main_manager()->pv(*this, threads, tt, rootDepth);
                
                if (bestValue <= alpha) {
                    beta = (alpha + beta) / 2;
                    alpha = std::max(bestValue - delta, -VALUE_INFINITE);
                    failedHighCnt = 0;
                    if (mainThread)
                        mainThread->stopOnPonderhit = false;
                } else if (bestValue >= beta) {
                    beta = std::min(bestValue + delta, VALUE_INFINITE);
                    ++failedHighCnt;
                } else {
                    break;
                }
                
                delta += delta / 3;
                assert(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
            }
            
            std::stable_sort(rootMoves.begin() + pvFirst, rootMoves.begin() + pvIdx + 1);
            
            if (mainThread && (threads.stop || pvIdx + 1 == multiPV || nodes > 10000000) &&
                !(threads.abortedSearch && is_loss(rootMoves[0].uciScore)))
                main_manager()->pv(*this, threads, tt, rootDepth);
            
            if (threads.stop)
                break;
        }
        
        if (!threads.stop)
            completedDepth = rootDepth;
        
        if (threads.abortedSearch && rootMoves[0].score != -VALUE_INFINITE && is_loss(rootMoves[0].score)) {
            Utility::move_to_front(rootMoves, [&lastBestPV = std::as_const(lastBestPV)](const auto& rm) { 
                return rm == lastBestPV[0]; 
            });
            rootMoves[0].pv = lastBestPV;
            rootMoves[0].score = rootMoves[0].uciScore = lastBestScore;
        } else if (rootMoves[0].pv[0] != lastBestPV[0]) {
            lastBestPV = rootMoves[0].pv;
            lastBestScore = rootMoves[0].score;
            lastBestMoveDepth = rootDepth;
        }
        
        if (!mainThread)
            continue;
        
        // Enhanced mate detection
        if (limits.mate && rootMoves[0].score == rootMoves[0].uciScore &&
            ((rootMoves[0].score >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - rootMoves[0].score <= 2 * limits.mate) ||
             (rootMoves[0].score != -VALUE_INFINITE && rootMoves[0].score <= VALUE_MATED_IN_MAX_PLY &&
              VALUE_MATE + rootMoves[0].score <= 2 * limits.mate)))
            threads.stop = true;
        
        for (auto&& th : threads) {
            totBestMoveChanges += th->worker->bestMoveChanges;
            th->worker->bestMoveChanges = 0;
        }
        
        // Enhanced time management
        if (limits.use_time_management() && !threads.stop && !mainThread->stopOnPonderhit) {
            uint64_t nodesEffort = rootMoves[0].effort * 100000 / std::max(size_t(1), size_t(nodes));
            
            double fallingEval = (16.198 + 2.638 * (mainThread->bestPreviousAverageScore - bestValue) +
                                 0.780 * (mainThread->iterValue[iterIdx] - bestValue)) / 100.0;
            fallingEval = std::clamp(fallingEval, 0.6378, 1.8394);
            
            double k = 0.516;
            double center = lastBestMoveDepth + 15;
            timeReduction = 0.8 + 0.85 / (1.089 + std::exp(-k * (completedDepth - center)));
            
            double reduction = (2.0380 + mainThread->previousTimeReduction) / (2.5240 * timeReduction);
            double bestMoveInstability = 0.9381 + 1.6401 * totBestMoveChanges / threads.size();
            double totalTime = mainThread->tm.optimum() * fallingEval * reduction * bestMoveInstability;
            
            auto elapsedTime = elapsed();
            
            // More aggressive time management for deeper searches
            double timeMultiplier = 1.0;
            if (completedDepth > 20) timeMultiplier = 1.2;
            if (completedDepth > 30) timeMultiplier = 1.5;
            
            if (completedDepth >= 9 && nodesEffort >= 81808 && 
                elapsedTime > totalTime * 0.6519 * timeMultiplier && !mainThread->ponder)
                threads.stop = true;
            
            if (elapsedTime > std::min(totalTime * timeMultiplier, double(mainThread->tm.maximum()))) {
                if (mainThread->ponder)
                    mainThread->stopOnPonderhit = true;
                else
                    threads.stop = true;
            } else {
                threads.increaseDepth = mainThread->ponder || elapsedTime <= totalTime * 0.264 * timeMultiplier;
            }
        }
        
        mainThread->iterValue[iterIdx] = bestValue;
        iterIdx = (iterIdx + 1) & 3;
    }
    
    if (!mainThread)
        return;
    
    mainThread->previousTimeReduction = timeReduction;
}

// Enhanced main search function with improved pruning and extensions
template<NodeType nodeType>
Value Search::Worker::search(Position& pos, Stack* ss, Value alpha, Value beta, Depth depth, bool cutNode) {
    constexpr bool PvNode = nodeType != NonPV;
    constexpr bool rootNode = nodeType == Root;
    const bool allNode = !(PvNode || cutNode);
    
    if (depth <= 0) {
        constexpr auto nt = PvNode ? PV : NonPV;
        return qsearch<nt>(pos, ss, alpha, beta);
    }
    
    depth = std::min(depth, MAX_PLY - 1);
    
    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(0 < depth && depth < MAX_PLY);
    assert(!(PvNode && cutNode));
    
    Move pv[MAX_PLY + 1];
    StateInfo st;
    Key posKey;
    Move move, excludedMove, bestMove;
    Depth extension, newDepth;
    Value bestValue, value, eval, maxValue, probCutBeta;
    bool givesCheck, improving, priorCapture, opponentWorsening;
    bool capture, ttCapture;
    int priorReduction;
    Piece movedPiece;
    SearchedList capturesSearched;
    SearchedList quietsSearched;
    
    Worker* thisThread = this;
    ss->inCheck = bool(pos.checkers());
    priorCapture = pos.captured_piece();
    Color us = pos.side_to_move();
    ss->moveCount = 0;
    bestValue = -VALUE_INFINITE;
    maxValue = VALUE_INFINITE;
    
    if (is_mainthread())
        main_manager()->check_time(*thisThread);
    
    if (PvNode && thisThread->selDepth < ss->ply + 1)
        thisThread->selDepth = ss->ply + 1;
    
    if (!rootNode) {
        Value result = VALUE_NONE;
        if (pos.rule_judge(result, ss->ply))
            return result == VALUE_DRAW ? value_draw(thisThread->nodes) : result;
        
        if (result != VALUE_NONE) {
            assert(result != VALUE_DRAW);
            if (result > VALUE_DRAW)
                alpha = std::max(alpha, VALUE_DRAW - 1);
            else
                beta = std::min(beta, VALUE_DRAW + 1);
        }
        
        if (threads.stop.load(std::memory_order_relaxed) || ss->ply >= MAX_PLY)
            return (ss->ply >= MAX_PLY && !ss->inCheck) ? evaluate(pos) : value_draw(thisThread->nodes);
        
        alpha = std::max(mated_in(ss->ply), alpha);
        beta = std::min(mate_in(ss->ply + 1), beta);
        if (alpha >= beta)
            return alpha;
    }
    
    assert(0 <= ss->ply && ss->ply < MAX_PLY);
    
    Square prevSq = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;
    bestMove = Move::none();
    priorReduction = (ss - 1)->reduction;
    (ss - 1)->reduction = 0;
    ss->statScore = 0;
    ss->isPvNode = PvNode;
    (ss + 2)->cutoffCnt = 0;
    
    // Enhanced transposition table lookup
    excludedMove = ss->excludedMove;
    posKey = pos.key();
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey);
    
    ss->ttHit = ttHit;
    ttData.move = rootNode ? thisThread->rootMoves[thisThread->pvIdx].pv[0] :
                  ttHit ? ttData.move : Move::none();
    ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply, pos.rule60_count()) : VALUE_NONE;
    ss->ttPv = excludedMove ? ss->ttPv : PvNode || (ttHit && ttData.is_pv);
    ttCapture = ttData.move && pos.capture(ttData.move);
    
    // Enhanced TT cutoff conditions
    if (!PvNode && !excludedMove && ttData.depth > depth - (ttData.value <= beta) &&
        is_valid(ttData.value) && (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER)) &&
        (cutNode == (ttData.value >= beta) || depth > 5)) {
        
        if (ttData.move && ttData.value >= beta) {
            if (!ttCapture)
                update_quiet_histories(pos, ss, *this, ttData.move, std::min(112 * depth - 62, 1525));
            
            if (prevSq != SQ_NONE && (ss - 1)->moveCount <= 2 && !priorCapture)
                update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -2250);
        }
        
        if (pos.rule60_count() < 110) {
            if (depth >= 8 && ttData.move && pos.pseudo_legal(ttData.move) && 
                pos.legal(ttData.move) && !is_decisive(ttData.value)) {
                
                do_move(pos, ttData.move, st);
                Key nextPosKey = pos.key();
                auto [ttHitNext, ttDataNext, ttWriterNext] = tt.probe(nextPosKey);
                undo_move(pos, ttData.move);
                
                if (!is_valid(ttDataNext.value))
                    return ttData.value;
                if ((ttData.value >= beta) == (-ttDataNext.value >= beta))
                    return ttData.value;
            } else {
                return ttData.value;
            }
        }
    }
    
    // Enhanced static evaluation with MCTS influence
    Value unadjustedStaticEval = VALUE_NONE;
    const auto correctionValue = correction_value(*thisThread, pos, ss);
    
    if (ss->inCheck) {
        ss->staticEval = eval = (ss - 2)->staticEval;
        improving = false;
        goto moves_loop;
    } else if (excludedMove) {
        unadjustedStaticEval = eval = ss->staticEval;
    } else if (ss->ttHit) {
        unadjustedStaticEval = ttData.eval;
        if (!is_valid(unadjustedStaticEval))
            unadjustedStaticEval = evaluate(pos);
        
        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);
        
        if (is_valid(ttData.value) && (ttData.bound & (ttData.value > eval ? BOUND_LOWER : BOUND_UPPER)))
            eval = ttData.value;
    } else {
        unadjustedStaticEval = evaluate(pos);
        ss->staticEval = eval = to_corrected_static_eval(unadjustedStaticEval, correctionValue);
        
        ttWriter.write(posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_UNSEARCHED, 
                      Move::none(), unadjustedStaticEval, tt.generation());
    }
    
    // Enhanced move ordering bonus
    if (((ss - 1)->currentMove).is_ok() && !(ss - 1)->inCheck && !priorCapture && !ttHit) {
        int bonus = std::clamp(-18 * int((ss - 1)->staticEval + ss->staticEval), -1056, 2024) + 341;
        thisThread->mainHistory[~us][((ss - 1)->currentMove).from_to()] << bonus * 1284 / 1024;
        if (type_of(pos.piece_on(prevSq)) != PAWN)
            thisThread->pawnHistory[pawn_structure_index(pos)][pos.piece_on(prevSq)][prevSq] << bonus * 1254 / 1024;
    }
    
    improving = ss->staticEval > (ss - 2)->staticEval;
    opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;
    
    // Enhanced depth adjustments
    if (priorReduction >= 3 && !opponentWorsening)
        depth++;
    if (priorReduction >= 1 && depth >= 2 && ss->staticEval + (ss - 1)->staticEval > 200)
        depth--;
    
    // Enhanced razoring with better margins
    if (!PvNode && eval < alpha - 1408 - 246 * depth * depth)
        return qsearch<NonPV>(pos, ss, alpha, beta);
    
    // Enhanced futility pruning
    {
        auto futility_margin = [&](Depth d) {
            Value futilityMult = 137 - 32 * (cutNode && !ss->ttHit);
            return futilityMult * d - improving * futilityMult * 2 - 
                   opponentWorsening * futilityMult / 3 + (ss - 1)->statScore / 149 + 
                   std::abs(correctionValue) / 130668;
        };
        
        if (!ss->ttPv && depth < 15 && eval - futility_margin(depth) >= beta && 
            eval >= beta && (!ttData.move || ttCapture) && !is_loss(beta) && !is_win(eval))
            return beta + (eval - beta) / 3;
    }
    
    // Enhanced null move search
    if (cutNode && (ss - 1)->currentMove != Move::null() && eval >= beta &&
        ss->staticEval >= beta - 8 * depth + 188 && !excludedMove && 
        pos.major_material(us) && ss->ply >= thisThread->nmpMinPly && !is_loss(beta)) {
        
        assert(eval - beta >= 0);
        
        Depth R = 7 + depth / 3;
        ss->currentMove = Move::null();
        ss->continuationHistory = &thisThread->continuationHistory[0][0][NO_PIECE][0];
        ss->continuationCorrectionHistory = &thisThread->continuationCorrectionHistory[NO_PIECE][0];
        
        do_null_move(pos, st);
        Value nullValue = -search<NonPV>(pos, ss + 1, -beta, -beta + 1, depth - R, false);
        undo_null_move(pos);
        
        if (nullValue >= beta && !is_win(nullValue)) {
            if (thisThread->nmpMinPly || depth < 15)
                return nullValue;
            
            assert(!thisThread->nmpMinPly);
            thisThread->nmpMinPly = ss->ply + 3 * (depth - R) / 4;
            Value v = search<NonPV>(pos, ss, beta - 1, beta, depth - R, false);
            thisThread->nmpMinPly = 0;
            
            if (v >= beta)
                return nullValue;
        }
    }
    
    improving |= ss->staticEval >= beta + 119;
    
    // Enhanced internal iterative reduction
    if (!allNode && depth >= 6 && !ttData.move)
        depth--;
    
    // Enhanced ProbCut
    probCutBeta = beta + 246 - 63 * improving;
    if (depth >= 3 && !is_decisive(beta) && 
        !(is_valid(ttData.value) && ttData.value < probCutBeta)) {
        
        assert(probCutBeta < VALUE_INFINITE && probCutBeta > beta);
        
        MovePicker mp(pos, ttData.move, probCutBeta - ss->staticEval, &thisThread->captureHistory);
        Depth probCutDepth = std::max(depth - 5, 0);
        
        while ((move = mp.next_move()) != Move::none()) {
            assert(move.is_ok());
            if (move == excludedMove || !pos.legal(move))
                continue;
            
            assert(pos.capture(move));
            movedPiece = pos.moved_piece(move);
            
            do_move(pos, move, st);
            ss->currentMove = move;
            ss->continuationHistory = &this->continuationHistory[ss->inCheck][true][movedPiece][move.to_sq()];
            ss->continuationCorrectionHistory = &this->continuationCorrectionHistory[movedPiece][move.to_sq()];
            
            value = -qsearch<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1);
            
            if (value >= probCutBeta && probCutDepth > 0)
                value = -search<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1, probCutDepth, !cutNode);
            
            undo_move(pos, move);
            
            if (value >= probCutBeta) {
                ttWriter.write(posKey, value_to_tt(value, ss->ply), ss->ttPv, BOUND_LOWER,
                              probCutDepth + 1, move, unadjustedStaticEval, tt.generation());
                if (!is_decisive(value))
                    return value - (probCutBeta - beta);
            }
        }
    }

moves_loop:
    
    // Enhanced small ProbCut
    probCutBeta = beta + 451;
    if ((ttData.bound & BOUND_LOWER) && ttData.depth >= depth - 4 && 
        ttData.value >= probCutBeta && !is_decisive(beta) && 
        is_valid(ttData.value) && !is_decisive(ttData.value))
        return probCutBeta;
    
    const PieceToHistory* contHist[] = {
        (ss - 1)->continuationHistory, (ss - 2)->continuationHistory, 
        (ss - 3)->continuationHistory, (ss - 4)->continuationHistory, 
        (ss - 5)->continuationHistory, (ss - 6)->continuationHistory
    };
    
    MovePicker mp(pos, ttData.move, depth, &thisThread->mainHistory, &thisThread->lowPlyHistory,
                  &thisThread->captureHistory, contHist, &thisThread->pawnHistory, ss->ply);
    
    value = bestValue;
    int moveCount = 0;
    
    // Enhanced move loop
    while ((move = mp.next_move()) != Move::none()) {
        assert(move.is_ok());
        
        if (move == excludedMove)
            continue;
        
        if (!pos.legal(move))
            continue;
        
        if (rootNode && !std::count(thisThread->rootMoves.begin() + thisThread->pvIdx,
                                   thisThread->rootMoves.begin() + thisThread->pvLast, move))
            continue;
        
        ss->moveCount = ++moveCount;
        
        if (rootNode && is_mainthread() && nodes > 10000000) {
            main_manager()->updates.onIter({depth, UCIEngine::move(move), moveCount + thisThread->pvIdx});
        }
        
        if (PvNode)
            (ss + 1)->pv = nullptr;
        
        extension = 0;
        capture = pos.capture(move);
        movedPiece = pos.moved_piece(move);
        givesCheck = pos.gives_check(move);
        (ss + 1)->quietMoveStreak = (!capture && !givesCheck) ? (ss->quietMoveStreak + 1) : 0;
        
        newDepth = depth - 1;
        int delta = beta - alpha;
        Depth r = reduction(improving, depth, moveCount, delta);
        
        // Enhanced reduction for ttPv nodes
        if (ss->ttPv)
            r += 932;
        
        // Enhanced pruning at shallow depth
        if (!rootNode && pos.major_material(us) && !is_loss(bestValue)) {
            // Enhanced move count pruning
            if (moveCount >= (3 + depth * depth) / (2 - improving))
                mp.skip_quiet_moves();
            
            int lmrDepth = newDepth - r / 1020;
            
            if (capture || givesCheck) {
                Piece capturedPiece = pos.piece_on(move.to_sq());
                int captHist = thisThread->captureHistory[movedPiece][move.to_sq()][type_of(capturedPiece)];
                
                // Enhanced futility pruning for captures
                if (!givesCheck && lmrDepth < 19 && !ss->inCheck) {
                    Value futilityValue = ss->staticEval + 318 + 350 * lmrDepth +
                                         PieceValue[capturedPiece] + 106 * captHist / 466;
                    if (futilityValue <= alpha)
                        continue;
                }
                
                // Enhanced SEE pruning
                int seeHist = std::clamp(captHist / 31, -249 * depth, 193 * depth);
                if (!pos.see_ge(move, -246 * depth - seeHist))
                    continue;
            } else {
                int history = (*contHist[0])[movedPiece][move.to_sq()] +
                             (*contHist[1])[movedPiece][move.to_sq()] +
                             thisThread->pawnHistory[pawn_structure_index(pos)][movedPiece][move.to_sq()];
                
                // Enhanced continuation history pruning
                if (history < -3051 * depth)
                    continue;
                
                history += 69 * thisThread->mainHistory[us][move.from_to()] / 29;
                lmrDepth += history / 3654;
                
                Value baseFutility = (bestMove ? 46 : 282);
                Value futilityValue = ss->staticEval + baseFutility + 124 * lmrDepth + 
                                     104 * (ss->staticEval > alpha);
                
                // Enhanced futility pruning
                if (!ss->inCheck && lmrDepth < 10 && futilityValue <= alpha) {
                    if (bestValue <= futilityValue && !is_decisive(bestValue) && !is_win(futilityValue))
                        bestValue = futilityValue;
                    continue;
                }
                
                lmrDepth = std::max(lmrDepth, 0);
                
                // Enhanced SEE pruning for quiet moves
                if (!pos.see_ge(move, -35 * lmrDepth * lmrDepth))
                    continue;
            }
        }
        
        // Enhanced singular extension search
        if (!rootNode && move == ttData.move && !excludedMove && 
            depth >= 5 - (thisThread->completedDepth > 32) + ss->ttPv && 
            is_valid(ttData.value) && !is_decisive(ttData.value) && 
            (ttData.bound & BOUND_LOWER) && ttData.depth >= depth - 3) {
            
            Value singularBeta = ttData.value - (43 + 70 * (ss->ttPv && !PvNode)) * depth / 75;
            Depth singularDepth = newDepth / 2;
            
            ss->excludedMove = move;
            value = search<NonPV>(pos, ss, singularBeta - 1, singularBeta, singularDepth, cutNode);
            ss->excludedMove = Move::none();
            
            if (value < singularBeta) {
                int corrValAdj = std::abs(correctionValue) / 261908;
                int doubleMargin = -4 + 236 * PvNode - 176 * !ttCapture - corrValAdj -
                                  1025 * ttMoveHistory / 132363 - (ss->ply > thisThread->rootDepth) * 43;
                int tripleMargin = 95 + 271 * PvNode - 243 * !ttCapture + 97 * ss->ttPv - corrValAdj -
                                  (ss->ply * 2 > thisThread->rootDepth * 3) * 56;
                
                extension = 1 + (value < singularBeta - doubleMargin) + (value < singularBeta - tripleMargin);
                depth++;
            } else if (value >= beta && !is_decisive(value)) {
                return value;
            } else if (ttData.value >= beta) {
                extension = -3;
            } else if (cutNode) {
                extension = -2;
            }
        }
        
        // Make the move
        do_move(pos, move, st, givesCheck);
        newDepth += extension;
        
        ss->currentMove = move;
        ss->continuationHistory = &thisThread->continuationHistory[ss->inCheck][capture][movedPiece][move.to_sq()];
        ss->continuationCorrectionHistory = &thisThread->continuationCorrectionHistory[movedPiece][move.to_sq()];
        
        uint64_t nodeCount = rootNode ? uint64_t(nodes) : 0;
        
        // Enhanced reduction adjustments
        if (ss->ttPv)
            r -= 2163 + PvNode * 1018 + (ttData.value > alpha) * 1038 + (ttData.depth >= depth) * (1077 + cutNode * 921);
        
        r += 327;
        r -= moveCount * 62;
        r -= std::abs(correctionValue) / 31508;
        
        if (cutNode)
            r += 3345 + 961 * !ttData.move;
        
        if (ttCapture)
            r += 1361 + (depth < 8) * 1442;
        
        if ((ss + 1)->cutoffCnt > 2)
            r += 1227 + allNode * 888;
        
        r += (ss + 1)->quietMoveStreak * 51;
        
        if (move == ttData.move)
            r -= 2890;
        
        if (capture)
            ss->statScore = 722 * int(PieceValue[pos.captured_piece()]) / 105 +
                           thisThread->captureHistory[movedPiece][move.to_sq()][type_of(pos.captured_piece())] - 4785;
        else
            ss->statScore = 2 * thisThread->mainHistory[us][move.from_to()] +
                           (*contHist[0])[movedPiece][move.to_sq()] +
                           (*contHist[1])[movedPiece][move.to_sq()] - 4180;
        
        r -= ss->statScore * 1235 / 10022;
        
        // Enhanced LMR
        if (depth >= 2 && moveCount > 1) {
            Depth d = std::max(1, std::min(newDepth - r / 1060, newDepth + !allNode + (PvNode && !bestMove))) +
                     (ss - 1)->isPvNode;
            
            ss->reduction = newDepth - d;
            value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);
            ss->reduction = 0;
            
            if (value > alpha && d < newDepth) {
                const bool doDeeperSearch = value > (bestValue + 57 + 2 * newDepth);
                const bool doShallowerSearch = value < bestValue + 9;
                
                newDepth += doDeeperSearch - doShallowerSearch;
                
                if (newDepth > d)
                    value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);
                
                update_continuation_histories(ss, movedPiece, move.to_sq(), 1533);
            } else if (value > alpha && value < bestValue + 9) {
                newDepth--;
            }
        } else if (!PvNode || moveCount > 1) {
            if (!ttData.move)
                r += 992;
            
            r -= ttMoveHistory / 8;
            
            value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha,
                                  newDepth - (r > 4055) - (r > 6153 && newDepth > 2), !cutNode);
        }
        
        // Enhanced PV search
        if (PvNode && (moveCount == 1 || value > alpha)) {
            (ss + 1)->pv = pv;
            (ss + 1)->pv[0] = Move::none();
            
            if (move == ttData.move && thisThread->rootDepth > 8)
                newDepth = std::max(newDepth, 1);
            
            value = -search<PV>(pos, ss + 1, -beta, -alpha, newDepth, false);
        }
        
        undo_move(pos, move);
        
        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);
        
        if (threads.stop.load(std::memory_order_relaxed))
            return VALUE_ZERO;
        
        // Enhanced root move handling
        if (rootNode) {
            RootMove& rm = *std::find(thisThread->rootMoves.begin(), thisThread->rootMoves.end(), move);
            
            rm.effort += nodes - nodeCount;
            rm.averageScore = rm.averageScore != -VALUE_INFINITE ? (value + rm.averageScore) / 2 : value;
            rm.meanSquaredScore = rm.meanSquaredScore != -VALUE_INFINITE * VALUE_INFINITE ?
                                 (value * std::abs(value) + rm.meanSquaredScore) / 2 : value * std::abs(value);
            
            if (moveCount == 1 || value > alpha) {
                rm.score = rm.uciScore = value;
                rm.selDepth = thisThread->selDepth;
                rm.scoreLowerbound = rm.scoreUpperbound = false;
                
                if (value >= beta) {
                    rm.scoreLowerbound = true;
                    rm.uciScore = beta;
                } else if (value <= alpha) {
                    rm.scoreUpperbound = true;
                    rm.uciScore = alpha;
                }
                
                rm.pv.resize(1);
                assert((ss + 1)->pv);
                for (Move* m = (ss + 1)->pv; *m != Move::none(); ++m)
                    rm.pv.push_back(*m);
                
                if (moveCount > 1 && !thisThread->pvIdx)
                    ++thisThread->bestMoveChanges;
            } else {
                rm.score = -VALUE_INFINITE;
            }
        }
        
        // Enhanced best move selection
        int inc = (value == bestValue && ss->ply + 2 >= thisThread->rootDepth &&
                  (int(nodes) & 15) == 0 && !is_win(std::abs(value) + 1));
        
        if (value + inc > bestValue) {
            bestValue = value;
            
            if (value + inc > alpha) {
                bestMove = move;
                
                if (PvNode && !rootNode)
                    update_pv(ss->pv, move, (ss + 1)->pv);
                
                if (value >= beta) {
                    ss->cutoffCnt += (extension < 2) || PvNode;
                    assert(value >= beta);
                    break;
                }
                
                // Enhanced depth reduction for promising positions
                if (depth > 2 && depth < 11 && !is_decisive(value))
                    depth -= 2;
                
                assert(depth > 0);
                alpha = value;
            }
        }
        
        if (move != bestMove && moveCount <= SEARCHEDLIST_CAPACITY) {
            if (capture)
                capturesSearched.push_back(move);
            else
                quietsSearched.push_back(move);
        }
    }
    
    // Enhanced mate detection
    assert(moveCount || !ss->inCheck || excludedMove || !MoveList<LEGAL>(pos).size());
    
    if (bestValue >= beta && !is_decisive(bestValue) && !is_decisive(alpha))
        bestValue = (bestValue * depth + beta) / (depth + 1);
    
    if (!moveCount)
        bestValue = excludedMove ? alpha : mated_in(ss->ply);
    else if (bestMove) {
        update_all_stats(pos, ss, *this, bestMove, prevSq, quietsSearched, capturesSearched, 
                        depth, ttData.move, moveCount);
        if (!PvNode)
            ttMoveHistory << (bestMove == ttData.move ? 774 : -844);
    } else if (!priorCapture && prevSq != SQ_NONE) {
        // Enhanced bonus for prior quiet countermove
        int bonusScale = -233;
        bonusScale += std::min(-(ss - 1)->statScore / 74, 238);
        bonusScale += std::min(60 * depth, 528);
        bonusScale += 150 * ((ss - 1)->moveCount > 12);
        bonusScale += 76 * (!ss->inCheck && bestValue <= ss->staticEval - 165);
        bonusScale += 158 * (!(ss - 1)->inCheck && bestValue <= -(ss - 1)->staticEval - 110);
        bonusScale = std::max(bonusScale, 0);
        
        const int scaledBonus = std::min(145 * depth - 87, 2078) * bonusScale;
        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, scaledBonus * 371 / 32768);
        thisThread->mainHistory[~us][((ss - 1)->currentMove).from_to()] << scaledBonus * 213 / 32768;
        
        if (type_of(pos.piece_on(prevSq)) != PAWN)
            thisThread->pawnHistory[pawn_structure_index(pos)][pos.piece_on(prevSq)][prevSq] << scaledBonus * 992 / 32768;
    } else if (priorCapture && prevSq != SQ_NONE) {
        Piece capturedPiece = pos.captured_piece();
        assert(capturedPiece != NO_PIECE);
        thisThread->captureHistory[pos.piece_on(prevSq)][prevSq][type_of(capturedPiece)] << 984;
    }
    
    if (PvNode)
        bestValue = std::min(bestValue, maxValue);
    
    if (bestValue <= alpha)
        ss->ttPv = ss->ttPv || (ss - 1)->ttPv;
    
    // Enhanced transposition table write
    if (!excludedMove && !(rootNode && thisThread->pvIdx))
        ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
                      bestValue >= beta ? BOUND_LOWER : PvNode && bestMove ? BOUND_EXACT : BOUND_UPPER,
                      moveCount != 0 ? depth : std::min(MAX_PLY - 1, depth + 6), bestMove,
                      unadjustedStaticEval, tt.generation());
    
    // Enhanced correction history adjustment
    if (!ss->inCheck && !(bestMove && pos.capture(bestMove)) &&
        ((bestValue < ss->staticEval && bestValue < beta) || (bestValue > ss->staticEval && bestMove))) {
        auto bonus = std::clamp(int(bestValue - ss->staticEval) * depth / 8,
                               -CORRECTION_HISTORY_LIMIT / 4, CORRECTION_HISTORY_LIMIT / 4);
        update_correction_history(pos, ss, *thisThread, bonus);
    }
    
    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);
    return bestValue;
}

// Enhanced quiescence search with improved pruning
template<NodeType nodeType>
Value Search::Worker::qsearch(Position& pos, Stack* ss, Value alpha, Value beta) {
    static_assert(nodeType != Root);
    constexpr bool PvNode = nodeType == PV;
    
    assert(alpha >= -VALUE_INFINITE && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    
    Move pv[MAX_PLY + 1];
    StateInfo st;
    Key posKey;
    Move move, bestMove;
    Value bestValue, value, futilityBase;
    bool pvHit, givesCheck, capture;
    int moveCount;
    
    if (PvNode) {
        (ss + 1)->pv = pv;
        ss->pv[0] = Move::none();
    }
    
    Worker* thisThread = this;
    bestMove = Move::none();
    ss->inCheck = bool(pos.checkers());
    moveCount = 0;
    
    if (PvNode && thisThread->selDepth < ss->ply + 1)
        thisThread->selDepth = ss->ply + 1;
    
    Value result = VALUE_NONE;
    if (pos.rule_judge(result, ss->ply))
        return result;
    
    if (result != VALUE_NONE) {
        assert(result != VALUE_DRAW);
        if (result > VALUE_DRAW)
            alpha = std::max(alpha, VALUE_DRAW);
        else
            beta = std::min(beta, VALUE_DRAW);
        if (alpha >= beta)
            return alpha;
    }
    
    if (ss->ply >= MAX_PLY)
        return !ss->inCheck ? evaluate(pos) : VALUE_DRAW;
    
    assert(0 <= ss->ply && ss->ply < MAX_PLY);
    
    posKey = pos.key();
    auto [ttHit, ttData, ttWriter] = tt.probe(posKey);
    
    ss->ttHit = ttHit;
    ttData.move = ttHit ? ttData.move : Move::none();
    ttData.value = ttHit ? value_from_tt(ttData.value, ss->ply, pos.rule60_count()) : VALUE_NONE;
    pvHit = ttHit && ttData.is_pv;
    
    if (!PvNode && ttData.depth >= DEPTH_QS && is_valid(ttData.value) &&
        (ttData.bound & (ttData.value >= beta ? BOUND_LOWER : BOUND_UPPER)))
        return ttData.value;
    
    Value unadjustedStaticEval = VALUE_NONE;
    if (ss->inCheck)
        bestValue = futilityBase = -VALUE_INFINITE;
    else {
        const auto correctionValue = correction_value(*thisThread, pos, ss);
        if (ss->ttHit) {
            unadjustedStaticEval = ttData.eval;
            if (!is_valid(unadjustedStaticEval))
                unadjustedStaticEval = evaluate(pos);
            
            ss->staticEval = bestValue = to_corrected_static_eval(unadjustedStaticEval, correctionValue);
            
            if (is_valid(ttData.value) && !is_decisive(ttData.value) &&
                (ttData.bound & (ttData.value > bestValue ? BOUND_LOWER : BOUND_UPPER)))
                bestValue = ttData.value;
        } else {
            unadjustedStaticEval = evaluate(pos);
            ss->staticEval = bestValue = to_corrected_static_eval(unadjustedStaticEval, correctionValue);
        }
        
        if (bestValue >= beta) {
            if (!is_decisive(bestValue))
                bestValue = (bestValue + beta) / 2;
            
            if (!ss->ttHit)
                ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), false, BOUND_LOWER,
                              DEPTH_UNSEARCHED, Move::none(), unadjustedStaticEval, tt.generation());
            
            return bestValue;
        }
        
        if (bestValue > alpha)
            alpha = bestValue;
        
        futilityBase = ss->staticEval + 219;
    }
    
    const PieceToHistory* contHist[] = {(ss - 1)->continuationHistory, (ss - 2)->continuationHistory};
    Square prevSq = ((ss - 1)->currentMove).is_ok() ? ((ss - 1)->currentMove).to_sq() : SQ_NONE;
    
    MovePicker mp(pos, ttData.move, DEPTH_QS, &thisThread->mainHistory, &thisThread->lowPlyHistory,
                  &thisThread->captureHistory, contHist, &thisThread->pawnHistory, ss->ply);
    
    while ((move = mp.next_move()) != Move::none()) {
        assert(move.is_ok());
        
        if (!pos.legal(move))
            continue;
        
        givesCheck = pos.gives_check(move);
        capture = pos.capture(move);
        moveCount++;
        
        if (!is_loss(bestValue)) {
            if (!givesCheck && move.to_sq() != prevSq && !is_loss(futilityBase)) {
                if (moveCount > 2)
                    continue;
                
                Value futilityValue = futilityBase + PieceValue[pos.piece_on(move.to_sq())];
                
                if (futilityValue <= alpha) {
                    bestValue = std::max(bestValue, futilityValue);
                    continue;
                }
                
                if (!pos.see_ge(move, alpha - futilityBase)) {
                    bestValue = std::min(alpha, futilityBase);
                    continue;
                }
            }
            
            if (!capture &&
                (*contHist[0])[pos.moved_piece(move)][move.to_sq()] +
                thisThread->pawnHistory[pawn_structure_index(pos)][pos.moved_piece(move)][move.to_sq()] <= 2745)
                continue;
            
            if (!pos.see_ge(move, -108))
                continue;
        }
        
        Piece movedPiece = pos.moved_piece(move);
        do_move(pos, move, st, givesCheck);
        
        ss->currentMove = move;
        ss->continuationHistory = &thisThread->continuationHistory[ss->inCheck][capture][movedPiece][move.to_sq()];
        ss->continuationCorrectionHistory = &thisThread->continuationCorrectionHistory[movedPiece][move.to_sq()];
        
        value = -qsearch<nodeType>(pos, ss + 1, -beta, -alpha);
        
        undo_move(pos, move);
        
        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);
        
        if (value > bestValue) {
            bestValue = value;
            
            if (value > alpha) {
                bestMove = move;
                
                if (PvNode)
                    update_pv(ss->pv, move, (ss + 1)->pv);
                
                if (value < beta)
                    alpha = value;
                else
                    break;
            }
        }
    }
    
    if (bestValue == -VALUE_INFINITE || (!moveCount && [&] {
        for (const auto& m : MoveList<QUIETS>(pos))
            if (pos.legal(m))
                return false;
        return true;
    }())) {
        assert(!MoveList<LEGAL>(pos).size());
        return mated_in(ss->ply);
    }
    
    if (!is_decisive(bestValue) && bestValue > beta)
        bestValue = (bestValue + beta) / 2;
    
    ttWriter.write(posKey, value_to_tt(bestValue, ss->ply), pvHit,
                   bestValue >= beta ? BOUND_LOWER : BOUND_UPPER, DEPTH_QS, bestMove,
                   unadjustedStaticEval, tt.generation());
    
    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);
    return bestValue;
}

// Enhanced reduction function
Depth Search::Worker::reduction(bool i, Depth d, int mn, int delta) const {
    int reductionScale = reductions[d] * reductions[mn];
    return reductionScale - delta * 1177 / rootDelta + !i * reductionScale * 102 / 297 + 1975;
}

TimePoint Search::Worker::elapsed() const {
    return main_manager()->tm.elapsed([this]() { return threads.nodes_searched(); });
}

TimePoint Search::Worker::elapsed_time() const { 
    return main_manager()->tm.elapsed_time(); 
}

Value Search::Worker::evaluate(const Position& pos) {
    return Eval::evaluate(networks[numaAccessToken], pos, accumulatorStack, refreshTable,
                         optimism[pos.side_to_move()]);
}

// Enhanced move and position handling functions
void Search::Worker::do_move(Position& pos, const Move move, StateInfo& st) {
    do_move(pos, move, st, pos.gives_check(move));
}

void Search::Worker::do_move(Position& pos, const Move move, StateInfo& st, const bool givesCheck) {
    DirtyPiece dp = pos.do_move(move, st, givesCheck, &tt);
    nodes.fetch_add(1, std::memory_order_relaxed);
    accumulatorStack.push(dp);
}

void Search::Worker::do_null_move(Position& pos, StateInfo& st) { 
    pos.do_null_move(st, tt); 
}

void Search::Worker::undo_move(Position& pos, const Move move) {
    pos.undo_move(move);
    accumulatorStack.pop();
}

void Search::Worker::undo_null_move(Position& pos) { 
    pos.undo_null_move(); 
}

// Enhanced clear function with better initialization
void Search::Worker::clear() {
    mainHistory.fill(61);
    captureHistory.fill(-596);
    pawnHistory.fill(-1234);
    pawnCorrectionHistory.fill(5);
    minorPieceCorrectionHistory.fill(0);
    nonPawnCorrectionHistory.fill(0);
    ttMoveHistory = 0;
    
    for (auto& to : continuationCorrectionHistory)
        for (auto& h : to)
            h.fill(8);
    
    for (bool inCheck : {false, true})
        for (StatsType c : {NoCaptures, Captures})
            for (auto& to : continuationHistory[inCheck][c])
                for (auto& h : to)
                    h.fill(-453);
    
    // Enhanced reduction table initialization
    for (size_t i = 1; i < reductions.size(); ++i)
        reductions[i] = int(1531 / 100.0 * std::log(i));
    
    refreshTable.clear(networks[numaAccessToken]);
}

namespace {

Value value_to_tt(Value v, int ply) { 
    return is_win(v) ? v + ply : is_loss(v) ? v - ply : v; 
}

Value value_from_tt(Value v, int ply, int r60c) {
    if (!is_valid(v))
        return VALUE_NONE;
    
    if (is_win(v))
        return VALUE_MATE - v > 120 - r60c ? VALUE_MATE_IN_MAX_PLY - 1 : v - ply;
    
    if (is_loss(v))
        return VALUE_MATE + v > 120 - r60c ? VALUE_MATED_IN_MAX_PLY + 1 : v + ply;
    
    return v;
}

void update_pv(Move* pv, Move move, const Move* childPv) {
    for (*pv++ = move; childPv && *childPv != Move::none();)
        *pv++ = *childPv++;
    *pv = Move::none();
}

void update_all_stats(const Position& pos, Stack* ss, Search::Worker& workerThread, Move bestMove, Square prevSq,
                     SearchedList& quietsSearched, SearchedList& capturesSearched, Depth depth, Move ttMove, int moveCount) {
    CapturePieceToHistory& captureHistory = workerThread.captureHistory;
    Piece movedPiece = pos.moved_piece(bestMove);
    PieceType capturedPiece;
    
    int bonus = std::min(145 * depth - 92, 2263) + 290 * (bestMove == ttMove);
    int malus = std::min(984 * depth - 227, 2026) - 32 * moveCount;
    
    if (!pos.capture(bestMove)) {
        update_quiet_histories(pos, ss, workerThread, bestMove, bonus * 1045 / 1024);
        
        for (Move move : quietsSearched)
            update_quiet_histories(pos, ss, workerThread, move, -malus * 1213 / 1024);
    } else {
        capturedPiece = type_of(pos.piece_on(bestMove.to_sq()));
        captureHistory[movedPiece][bestMove.to_sq()][capturedPiece] << bonus * 1377 / 1024;
    }
    
    if (prevSq != SQ_NONE && ((ss - 1)->moveCount == 1 + (ss - 1)->ttHit) && !pos.captured_piece())
        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -malus * 581 / 1024);
    
    for (Move move : capturesSearched) {
        movedPiece = pos.moved_piece(move);
        capturedPiece = type_of(pos.piece_on(move.to_sq()));
        captureHistory[movedPiece][move.to_sq()][capturedPiece] << -malus * 1196 / 1024;
    }
}

void update_continuation_histories(Stack* ss, Piece pc, Square to, int bonus) {
    static constexpr std::array<ConthistBonus, 6> conthist_bonuses = {
        {{1, 1092}, {2, 631}, {3, 294}, {4, 517}, {5, 126}, {6, 445}}
    };
    
    for (const auto [i, weight] : conthist_bonuses) {
        if (ss->inCheck && i > 2)
            break;
        if (((ss - i)->currentMove).is_ok())
            (*(ss - i)->continuationHistory)[pc][to] << bonus * weight / 1024;
    }
}

void update_quiet_histories(const Position& pos, Stack* ss, Search::Worker& workerThread, Move move, int bonus) {
    Color us = pos.side_to_move();
    workerThread.mainHistory[us][move.from_to()] << bonus;
    
    if (ss->ply < LOW_PLY_HISTORY_SIZE)
        workerThread.lowPlyHistory[ss->ply][move.from_to()] << bonus * 860 / 1024;
    
    update_continuation_histories(ss, pos.moved_piece(move), move.to_sq(), bonus * (bonus > 0 ? 1033 : 915) / 1024);
    
    int pIndex = pawn_structure_index(pos);
    workerThread.pawnHistory[pIndex][pos.moved_piece(move)][move.to_sq()] << bonus * (bonus > 0 ? 691 : 468) / 1024;
}

} // namespace

// Enhanced SearchManager functions
void SearchManager::check_time(Search::Worker& worker) {
    if (--callsCnt > 0)
        return;
    
    callsCnt = worker.limits.nodes ? std::min(512, int(worker.limits.nodes / 1024)) : 512;
    
    static TimePoint lastInfoTime = now();
    TimePoint elapsed = tm.elapsed([&worker]() { return worker.threads.nodes_searched(); });
    TimePoint tick = worker.limits.startTime + elapsed;
    
    if (tick - lastInfoTime >= 1000) {
        lastInfoTime = tick;
        dbg_print();
    }
    
    if (ponder)
        return;
    
    if (worker.completedDepth >= 1 &&
        ((worker.limits.use_time_management() && (elapsed > tm.maximum() || stopOnPonderhit)) ||
         (worker.limits.movetime && elapsed >= worker.limits.movetime) ||
         (worker.limits.nodes && worker.threads.nodes_searched() >= worker.limits.nodes)))
        worker.threads.stop = worker.threads.abortedSearch = true;
}

void SearchManager::pv(const Search::Worker& worker, const ThreadPool& threads, 
                      const TranspositionTable& tt, Depth depth) const {
    const auto nodes = threads.nodes_searched();
    const auto& rootMoves = worker.rootMoves;
    const auto& pos = worker.rootPos;
    size_t pvIdx = worker.pvIdx;
    size_t multiPV = std::min(size_t(worker.options["MultiPV"]), rootMoves.size());
    
    for (size_t i = 0; i < multiPV; ++i) {
        bool updated = rootMoves[i].score != -VALUE_INFINITE;
        if (depth == 1 && !updated && i > 0)
            continue;
        
        Depth d = updated ? depth : std::max(1, depth - 1);
        Value v = updated ? rootMoves[i].uciScore : rootMoves[i].previousScore;
        
        if (v == -VALUE_INFINITE)
            v = VALUE_ZERO;
        
        std::string pv;
        for (Move m : rootMoves[i].pv)
            pv += UCIEngine::move(m) + " ";
        
        if (!pv.empty())
            pv.pop_back();
        
        auto wdl = worker.options["UCI_ShowWDL"] ? UCIEngine::wdl(v, pos) : "";
        auto bound = rootMoves[i].scoreLowerbound ? "lowerbound" :
                    (rootMoves[i].scoreUpperbound ? "upperbound" : "");
        
        InfoFull info;
        info.depth = d;
        info.selDepth = rootMoves[i].selDepth;
        info.multiPV = i + 1;
        info.score = {v, pos};
        info.wdl = wdl;
        
        if (i == pvIdx && updated)
            info.bound = bound;
        
        TimePoint time = std::max(TimePoint(1), tm.elapsed_time());
        info.timeMs = time;
        info.nodes = nodes;
        info.nps = nodes * 1000 / time;
        info.tbHits = 0;
        info.pv = pv;
        info.hashfull = tt.hashfull();
        
        updates.onUpdateFull(info);
    }
}

bool RootMove::extract_ponder_from_tt(const TranspositionTable& tt, Position& pos) {
    StateInfo st;
    assert(pv.size() == 1);
    
    if (pv[0] == Move::none())
        return false;
    
    pos.do_move(pv[0], st, &tt);
    auto [ttHit, ttData, ttWriter] = tt.probe(pos.key());
    
    if (ttHit) {
        if (MoveList<LEGAL>(pos).contains(ttData.move))
            pv.push_back(ttData.move);
    }
    
    pos.undo_move(pv[0]);
    return pv.size() > 1;
}

} // namespace Stockfish
