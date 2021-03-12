use hnfen::{
    moves::Move,
    types::{Board, Hnfen, Player},
};
use hnqai::{load_trainers, BoardState, HnefataflTerminator, SAVE_FILE};
use rurel::strategy::terminate::TerminationStrategy;
use std::io::{self, Write};

fn query(msg: &str) -> String {
    print!("{}", msg);
    print!(" ");
    io::stdout().flush().ok();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_string()
}

fn main() {
    println!("Starting HNQAI Command Line Interface!");

    println!("Loading trainers into memory");
    let trainers = load_trainers(SAVE_FILE);
    println!("Done loading");

    loop {
        let input = query("Use default board? [Y/n]");
        let mut board = match input.as_str() {
            "N" | "n" => {
                let input = query("Enter board hnfen:");
                let board = if let Some(b) = Board::from_hnfen(input.as_str()) {
                    b
                } else {
                    continue;
                };
                BoardState {
                    player: board.next,
                    board,
                    num_moves: 0,
                }
            }
            _ => {
                println!("Using default board");
                BoardState::default()
            }
        };

        let mut terminator = HnefataflTerminator::new(1000);

        let input = query("Is the human player (a)ttacker or (d)efender? [a/d]");
        let player = match input.as_str() {
            "a" | "A" => Player::Black,
            "d" | "D" => Player::White,
            _ => {
                println!("Unknown player");
                continue;
            }
        };

        println!("Starting game");
        println!("{}", board.board.pretty());

        loop {
            if terminator.should_stop(&board) {
                break;
            }
            let turn = board.board.next;
            match () {
                _ if turn == player => {
                    // Players turn
                    let input = query("Human move:");
                    match input.as_str() {
                        "q" | "quit" => break,
                        _ => {}
                    }
                    let p_move = if let Some(m) = Move::from_hnfen(input.as_str()) {
                        m
                    } else {
                        println!("Interpreting move failed, try again.");
                        continue;
                    };
                    board.board.apply(&p_move);
                    println!("Player's move: {}\n{}", input, board.board.pretty());
                }
                _ => {
                    // CPUs turn
                    if let Some(cpu_move) = trainers.apply_best(&mut board) {
                        println!(
                            "CPU's move: {}\n{}",
                            cpu_move.piece_move,
                            board.board.pretty()
                        );
                    }
                }
            }
        }

        println!(
            "Game over: {:?} won",
            if board.board.king_escaped() {
                Player::White
            } else {
                Player::Black
            }
        );

        let input = query("Quit?: [q]");
        match input.as_str() {
            "Q" | "q" | "quit" => break,
            _ => {}
        }
    }

    /*
    struct Rewards {
        attack: Vec<f64>,
        defense: Vec<f64>,
    }
    let mut rewards = Rewards {
        attack: Vec::new(),
        defense: Vec::new(),
    };

    for (i, b) in game.boards.into_iter().enumerate() {
        let attack_reward = {
            let mut b_b = b.clone();
            b_b.player = Player::Black;
            b_b.reward()
        };
        let defense_reward = {
            let mut b_b = b.clone();
            b_b.player = Player::White;
            b_b.reward()
        };
        println!(
            "{}\nattack reward = {}, defense reward = {}",
            b.board.pretty(),
            attack_reward,
            defense_reward,
        );
        rewards.attack.push(attack_reward);
        rewards.defense.push(defense_reward);
        if let Some(Some(m)) = game.moves.get(i) {
            println!("{}. {}", i + 1, m.piece_move);
        } else {
            println!("{}: No move", i + 1);
        }
    }

    use plotters::prelude::*;
    let img = BitMapBackend::new("reward_progression.png", (1024, 768)).into_drawing_area();
    img.fill(&WHITE).unwrap();
    let (min, max) = rewards
        .attack
        .iter()
        .chain(rewards.defense.iter())
        .fold((f64::MAX, f64::MIN), |(cmin, cmax), &x| {
            (cmin.min(x), cmax.max(x))
        });
    println!("Range: {} to {}", min, max);
    let mut chart = ChartBuilder::on(&img)
        .caption("Reward progression", ("Arial", 30))
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(0..rewards.attack.len(), (min - 1.0)..(max + 1.0))
        .unwrap();
    chart.configure_mesh().draw().unwrap();
    chart
        .draw_series(LineSeries::new(
            rewards.attack.into_iter().enumerate(),
            &RED,
        ))
        .unwrap();
    chart
        .draw_series(LineSeries::new(
            rewards.defense.into_iter().enumerate(),
            &BLUE,
        ))
        .unwrap();

    loop {
        apply_best_action(&mut board, Some(&trainers.attacker));
        println!("{}", board.board.pretty());
        if terminator.should_stop(&board) {
            break;
        }
        apply_best_action(&mut board, Some(&trainers.defender));
        println!("{}", board.board.pretty());
        if terminator.should_stop(&board) {
            break;
        }
    }
    */
}
