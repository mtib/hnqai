use std::{collections::HashMap, fs::File, io::BufReader};

use hnfen::{
    moves::{possible_moves, Move, Position},
    types::{Board, Player},
};
use rand::thread_rng;
use rurel::{mdp::State, strategy::terminate::TerminationStrategy, AgentTrainer};
use serde::{Deserialize, Serialize};

pub const SAVE_FILE: &str = "qstates.cbor";
pub const DEFAULT_MAX_GAME_LENGTH: isize = 1000;

#[derive(PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct BoardState {
    pub board: Board,
    pub player: Player,
    pub num_moves: usize,
}

impl Default for BoardState {
    fn default() -> Self {
        BoardState {
            board: Board::default(),
            num_moves: 0,
            player: Player::Black,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Serialize, Deserialize)]
pub struct BoardMove {
    pub piece_move: Move,
}

impl State for BoardState {
    type A = BoardMove;

    fn reward(&self) -> f64 {
        //let white_positions = self.board.pieces(Player::White);
        //let white_pieces = white_positions.len() as f64;
        let black_pieces = self.board.pieces(Player::Black).len() as f64;
        let king = self.board.king();
        let king_in_corner = self.board.king_escaped();

        const N_CLOSEST: usize = 10;

        let kd = if let Some(p) = king {
            let (kx, ky) = p.to_indices();
            let mut dists = self
                .board
                .pieces(Player::Black)
                .into_iter()
                .map(|p| {
                    let (x, y) = p.to_indices();
                    ((kx as isize) - (x as isize)).abs() + ((ky as isize) - (y as isize)).abs()
                })
                .collect::<Vec<isize>>();
            dists.sort_unstable();
            dists.iter().take(N_CLOSEST).sum()
        } else {
            0
        };

        match self.player {
            Player::Black => match () {
                _ if king_in_corner => -1000f64,
                _ if king.is_none() => 1000f64,
                _ => {
                    black_pieces
                        + ((N_CLOSEST.min(black_pieces as usize) * 10) as f64 - kd as f64) * 2f64
                        - (self.num_moves as f64) / 4f64
                }
            },
            Player::White => match () {
                _ if king_in_corner => 1000f64,
                _ if king.is_none() => -1000f64,
                _ => {
                    let k_pos = king.unwrap().to_indices();
                    // King's distance from the middle
                    let mhdist = (5 - k_pos.0 as isize).abs() + (5 - k_pos.1 as isize).abs();
                    mhdist as f64 * 10f64 - black_pieces - (self.num_moves as f64) / 4f64
                }
            },
        }
    }

    fn actions(&self) -> Vec<Self::A> {
        // Game is over, don't do anything anymore
        if self.board.king().is_none() || self.board.king_escaped() {
            return vec![BoardMove {
                piece_move: Move {
                    from: Position::from_indices(0, 0),
                    to: Position::from_indices(0, 0),
                },
            }];
        }

        let moves: Vec<BoardMove> = possible_moves(&self.board)
            .into_iter()
            .map(|x| BoardMove { piece_move: x })
            .collect();
        moves
    }
}

pub struct HnefataflTerminator(isize);

impl HnefataflTerminator {
    pub fn new(max_game_length: isize) -> Self {
        HnefataflTerminator(max_game_length)
    }
}

impl Default for HnefataflTerminator {
    fn default() -> Self {
        HnefataflTerminator::new(DEFAULT_MAX_GAME_LENGTH)
    }
}

impl TerminationStrategy<BoardState> for HnefataflTerminator {
    fn should_stop(&mut self, state: &BoardState) -> bool {
        self.0 -= 1;
        state.board.king().is_none()
            || state.board.king_escaped()
            || possible_moves(&state.board).is_empty()
            || self.0 <= 0
    }
}

#[derive(Serialize, Deserialize)]
pub struct AIs {
    pub attacker: HashMap<BoardState, HashMap<BoardMove, f64>>,
    pub defender: HashMap<BoardState, HashMap<BoardMove, f64>>,
}

pub struct Trainers {
    pub attacker: AgentTrainer<BoardState>,
    pub defender: AgentTrainer<BoardState>,
}

impl Default for Trainers {
    fn default() -> Self {
        Trainers {
            attacker: AgentTrainer::new(),
            defender: AgentTrainer::new(),
        }
    }
}

pub struct Stats {
    pub boards: Vec<BoardState>,
    pub moves: Vec<Option<BoardMove>>,
}

impl Default for Stats {
    fn default() -> Self {
        Stats {
            boards: Vec::new(),
            moves: Vec::new(),
        }
    }
}

impl Trainers {
    pub fn run_best_game(&self) -> Stats {
        let mut board = BoardState::default();
        let mut terminator = HnefataflTerminator::default();
        let mut stats = Stats::default();
        stats.boards.push(board.clone());
        loop {
            stats
                .moves
                .push(apply_best_action(&mut board, Some(&self.attacker)));
            stats.boards.push(board.clone());
            if terminator.should_stop(&board) {
                break;
            }
            stats
                .moves
                .push(apply_best_action(&mut board, Some(&self.defender)));
            stats.boards.push(board.clone());
            if terminator.should_stop(&board) {
                break;
            }
        }
        stats
    }

    pub fn apply_best(&self, board: &mut BoardState) -> Option<BoardMove> {
        match board.board.next {
            Player::Black => apply_best_action(board, Some(&self.attacker)),
            Player::White => apply_best_action(board, Some(&self.defender)),
        }
    }
}

pub fn load_state(path: &str) -> Option<AIs> {
    let save = File::open(path).ok()?;
    let reader = BufReader::new(save);
    serde_cbor::from_reader::<AIs, _>(reader).ok()
}

pub fn load_trainers(path: &str) -> Trainers {
    if let Some(state) = load_state(path) {
        let mut attack_trainer = AgentTrainer::new();
        let mut defense_trainer = AgentTrainer::new();
        attack_trainer.import_state(state.attacker);
        defense_trainer.import_state(state.defender);

        Trainers {
            attacker: attack_trainer,
            defender: defense_trainer,
        }
    } else {
        println!("Loading trainers failed, returning new trainers!");
        Trainers::default()
    }
}

pub fn apply_best_action(
    state: &mut BoardState,
    trainer: Option<&AgentTrainer<BoardState>>,
) -> Option<BoardMove> {
    use rand::seq::SliceRandom;
    let mut rng = thread_rng();

    let moves: Vec<_> = possible_moves(&state.board)
        .into_iter()
        .map(|x| BoardMove { piece_move: x })
        .collect();
    if trainer.is_none() {
        let first_move = moves.choose(&mut rng).unwrap();
        state.board.apply(&first_move.piece_move);
        return Some(first_move.clone());
    }
    let mut best_move = None;
    for o_move in moves.into_iter() {
        if let Some(val) = trainer.unwrap().expected_value(&state, &o_move) {
            // Trainer has an opinion
            if let Some((ref mut v, best_so_far)) = best_move {
                // There already is a best move so far
                if val > best_so_far {
                    best_move = Some((vec![o_move], val));
                } else if (val - best_so_far).abs() < 0.01 {
                    v.push(o_move);
                }
            } else {
                // There wasn't a best move before (init)
                best_move = Some((vec![o_move], val));
            }
        } else {
            // Trainer doesn't have an opinion
            if let Some((ref mut v, val)) = best_move {
                // There was already some move
                if (val - f64::MIN).abs() < 0.01 {
                    v.push(o_move);
                }
            } else {
                // There were no moves before
                best_move = Some((vec![o_move], f64::MIN));
            }
        }
    }
    if let Some((best_moves, _)) = best_move {
        let chosen_move = best_moves.choose(&mut rng).unwrap();
        state.board.apply(&chosen_move.piece_move);
        return Some(chosen_move.clone());
    };
    None
}
