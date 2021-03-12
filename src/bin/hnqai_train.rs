use std::{fs::File, io::BufWriter};

use hnqai::{
    apply_best_action, load_trainers, AIs, BoardMove, BoardState, HnefataflTerminator, SAVE_FILE,
};
use rurel::{
    mdp::Agent,
    strategy::{explore::RandomExploration, learn::QLearning},
    AgentTrainer,
};

struct HnAgent<'a> {
    state: BoardState,
    opponent: Option<&'a AgentTrainer<BoardState>>,
}

impl<'a> Agent<BoardState> for HnAgent<'a> {
    fn current_state(&self) -> &BoardState {
        &self.state
    }

    fn take_action(&mut self, action: &BoardMove) {
        self.state.board.apply(&action.piece_move);
        self.state.num_moves += 1;
        self.state.player = self.state.player.opposite();
        apply_best_action(&mut self.state, self.opponent);
        self.state.num_moves += 1;
    }
}

const EPOCHS: usize = 50;
const DOUBLE_GAMES_PER_EPOCH: usize = 50;
const MAX_GAME_LENGTH: isize = 5000;
const GAMMA: f64 = 1.0 - 1.0 / 50.0; // Usual games are around 50 rounds
const ALPHA: f64 = 0.6; // There aren't that many *random* responses
const INITIAL: f64 = 2.; // IDK?

fn main() {
    let trainers = load_trainers(SAVE_FILE);
    let mut attack_trainer = trainers.attacker;
    let mut defense_trainer = trainers.defender;

    let start_state = BoardState::default();
    let mut escape_wins = 0;
    let mut caputre_wins = 0;
    let mut draws = 0;

    for epoch in 1..=EPOCHS {
        for _ in 1..=DOUBLE_GAMES_PER_EPOCH {
            let mut attack_agent = HnAgent {
                state: start_state.clone(),
                opponent: Some(&defense_trainer),
            };
            attack_trainer.train(
                &mut attack_agent,
                &QLearning::new(ALPHA, GAMMA, INITIAL),
                &mut HnefataflTerminator::new(MAX_GAME_LENGTH),
                &RandomExploration::new(),
            );
            if attack_agent.state.board.king().is_none() {
                caputre_wins += 1;
            } else if attack_agent.state.board.king_escaped() {
                escape_wins += 1;
            } else {
                draws += 1;
            }
        }

        for _ in 1..=DOUBLE_GAMES_PER_EPOCH {
            let mut start_state = start_state.clone();
            apply_best_action(&mut start_state, Some(&attack_trainer));
            let mut defense_agent = HnAgent {
                state: start_state,
                opponent: Some(&attack_trainer),
            };
            defense_trainer.train(
                &mut defense_agent,
                &QLearning::new(ALPHA, GAMMA, INITIAL),
                &mut HnefataflTerminator::new(MAX_GAME_LENGTH),
                &RandomExploration::new(),
            );
            if defense_agent.state.board.king().is_none() {
                caputre_wins += 1;
            } else if defense_agent.state.board.king_escaped() {
                escape_wins += 1;
            } else {
                draws += 1;
            }
        }
        println!(
            "Epoch {} ({} total games): {} wins by escape, {} wins by attack, {} draws",
            epoch,
            epoch * 2 * DOUBLE_GAMES_PER_EPOCH,
            escape_wins,
            caputre_wins,
            draws
        );
    }

    let save = File::create(SAVE_FILE).unwrap();
    let writer = BufWriter::new(save);
    serde_cbor::to_writer(
        writer,
        &AIs {
            attacker: attack_trainer.export_state(),
            defender: defense_trainer.export_state(),
        },
    )
    .ok();
}
