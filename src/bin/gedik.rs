use gedik::board::Position;
use gedik::heuristics::{candidate_moves, playout, Rng};
use gedik::engine::Engine;
use gedik::mcts::Mcts;
use gedik::notation::{board_string, move_name, parse_move, to_fen};
use gedik::perft::{divide, perft};
use gedik::search::Searcher;
use std::io::{self, BufRead, Write};
use std::time::Instant;

const MAX_PLY: u16 = 400;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    match cmd {
        "perft" => cmd_perft(&args),
        "divide" => cmd_divide(&args),
        "bench" => cmd_bench(),
        "selfplay" => cmd_selfplay(&args),
        "match" => cmd_match(&args),
        "tournament" | "turnuva" => cmd_tournament(&args),
        "play" => cmd_play(&args),
        "serve" | "web" => cmd_serve(&args),
        _ => help(),
    }
}

fn help() {
    println!(
        "quoridor — bitboard Quoridor motoru\n\n\
         KOMUTLAR\n\
         \x20 play [taraf] [motor]        insan vs motor (taraf 0 = alttan başlarsın)\n\
         \x20 selfplay [motor]            motor kendine karşı\n\
         \x20 match <A> <B> [açılış] [seed] [ply]  iki motoru karşılaştır (her açılış 2 renk)\n\
         \x20 tournament <açılış> <motor...>       round-robin turnuva + Elo, ratings.json yazar\n\
         \x20 perft <derinlik>            düğüm sayımı\n\
         \x20 divide <derinlik>           kök hamle başına dağılım\n\
         \x20 serve [port]                tarayıcı arayüzü (varsayılan 8080)\n\
         \x20 bench                       hız ölçümü\n\n\
         MOTOR TANIMI\n\
         \x20 mcts:1000ms   süre bütçeli MCTS\n\
         \x20 mcts:20000    sabit rollout bütçeli MCTS (gorisanson seviyeleri:\n\
         \x20               2500 novice · 7500 average · 20000 good · 60000 strong)\n\
         \x20 ab:500ms      baseline alpha-beta\n\n\
         NOTASYON\n\
         \x20 e2   piyonu o kareye oyna · e6h yatay duvar · c3v dikey duvar\n\
         \x20 moves   legal hamleleri listele · quit  çık\n"
    );
}

fn num<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn arg(args: &[String], i: usize, default: &str) -> String {
    args.get(i).cloned().unwrap_or_else(|| default.to_string())
}

fn cmd_perft(args: &[String]) {
    let depth: u32 = num(args, 2, 3);
    let mut pos = Position::start();
    for d in 1..=depth {
        let t = Instant::now();
        let n = perft(&mut pos, d);
        let el = t.elapsed();
        println!(
            "perft({d}) = {n:>14}   {:>8.3} s   {:>12.0} düğüm/s",
            el.as_secs_f64(),
            n as f64 / el.as_secs_f64().max(1e-9)
        );
    }
}

fn cmd_divide(args: &[String]) {
    let depth: u32 = num(args, 2, 2);
    let mut pos = Position::start();
    let mut rows = divide(&mut pos, depth);
    rows.sort_by_key(|&(m, _)| m.0);
    let mut total = 0u64;
    for (mv, n) in &rows {
        println!("{:>5} {:>12}", move_name(&pos, *mv), n);
        total += n;
    }
    println!("{} hamle, toplam {}", rows.len(), total);
}

fn cmd_bench() {
    let pos = Position::start();

    let t = Instant::now();
    let iters = 200_000;
    let mut count = 0u64;
    for _ in 0..iters {
        count += pos.legal_moves().len() as u64;
    }
    println!(
        "hamle üretimi : {:>12.0} pozisyon/s   (başlangıçta {} legal hamle)",
        iters as f64 / t.elapsed().as_secs_f64(),
        count / iters
    );

    let m = pos.masks();
    let t = Instant::now();
    let mut acc = 0u128;
    for _ in 0..1_000_000 {
        acc ^= m.reachable(40);
    }
    println!(
        "flood-fill    : {:>12.0} kez/s        (sağlama {})",
        1_000_000.0 / t.elapsed().as_secs_f64(),
        acc.count_ones()
    );

    let mut rng = Rng::new(0x1234_5678);
    let t = Instant::now();
    let n = 20_000;
    let mut wins = 0usize;
    for _ in 0..n {
        wins += playout(pos, &mut rng, 0.30, 400);
    }
    println!(
        "rollout       : {:>12.0} rollout/s    (B kazanma oranı {:.3})",
        n as f64 / t.elapsed().as_secs_f64(),
        wins as f64 / n as f64
    );

    let mut mc = Mcts::new(1);
    let (mv, st) = mc.search_time(&pos, 3000);
    println!(
        "mcts (3 sn)   : {:>12.0} rollout/s    en iyi {} wr {:.3} ağaç {} düğüm",
        st.rollouts as f64 / st.elapsed_s,
        mv.map(|m| move_name(&pos, m)).unwrap_or("-".into()),
        st.win_rate,
        st.nodes
    );

    let mut s = Searcher::new(20);
    let t = Instant::now();
    let (mv, sc) = s.best_move(&pos, 12, 3000);
    println!(
        "ab (3 sn)     : {:>12.0} düğüm/s      en iyi {} ev {}",
        s.nodes as f64 / t.elapsed().as_secs_f64(),
        mv.map(|m| move_name(&pos, m)).unwrap_or("-".into()),
        sc
    );
}

fn cmd_selfplay(args: &[String]) {
    let spec = arg(args, 2, "mcts:300ms");
    let spec_b = arg(args, 3, &spec);
    let mut e0 = Engine::parse(&spec, 11);
    let mut e1 = Engine::parse(&spec_b, 22);
    let mut pos = Position::start();
    let mut history: Vec<String> = Vec::new();

    while pos.winner().is_none() && pos.ply < MAX_PLY {
        let e = if pos.side == 0 { &mut e0 } else { &mut e1 };
        let c = e.choose(&pos);
        let (mv, info, work) = (c.mv, c.info, c.work);
        let Some(mv) = mv else { break };
        let name = move_name(&pos, mv);
        history.push(name.clone());
        println!(
            "{:>3}. {} {:<5} {}  iş {}",
            pos.ply / 2 + 1,
            if pos.side == 0 { "A" } else { "B" },
            name,
            info,
            work
        );
        pos.make(mv);
    }

    println!("\n{}", board_string(&pos));
    match pos.winner() {
        Some(w) => println!(
            "Kazanan: {} ({} yarım hamlede)",
            if w == 0 { "A" } else { "B" },
            pos.ply
        ),
        None => println!("Hamle sınırı ({} yarım hamle)", pos.ply),
    }
    println!("Oyun: {}", history.join(" "));
}

/// Rastgele ama makul bir açılış pozisyonu.
///
/// Değerlendirme yaprağıyla çalışan MCTS **deterministik**, dolayısıyla
/// başlangıç pozisyonundan oynanan N maç aslında tek maç. Anlamlı bir örneklem
/// için her maç farklı bir açılıştan başlıyor ve her açılış renkler
/// değiştirilerek iki kez oynanıyor — böylece dengesiz açılışlar ortalamada
/// birbirini götürüyor.
fn random_opening(seed: u64, plies: usize) -> Position {
    let mut rng = Rng::new(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    'outer: loop {
        let mut p = Position::start();
        for _ in 0..plies {
            if p.winner().is_some() {
                continue 'outer;
            }
            // Tamamen rastgele değil: aday hamleler, böylece pozisyon saçma olmuyor.
            let moves = candidate_moves(&p);
            if moves.is_empty() {
                continue 'outer;
            }
            let mv = moves[rng.below(moves.len())];
            p.make(mv);
        }
        if p.winner().is_none() {
            return p;
        }
    }
}

/// Canlı izleme: `GEDIK_CANLI` ortam değişkeni bir dosya yolu ise, oynanan
/// oyunun hali her hamleden sonra o dosyaya JSON olarak yazılır. Web
/// arayüzü (`/api/canli`) `deney/*.canli` dosyalarını okuyup tahtada
/// gösterir. Değişken yoksa hiçbir şey yazılmaz, maç eskisi gibi.
struct Canli {
    yol: String,
    bas: String,
}

impl Canli {
    fn yeni(spec_a: &str, spec_b: &str) -> Option<Canli> {
        let yol = std::env::var("GEDIK_CANLI").ok().filter(|y| !y.is_empty())?;
        let bas = format!(
            r#""a":"{}","b":"{}""#,
            json_kac(spec_a),
            json_kac(spec_b)
        );
        Some(Canli { yol, bas })
    }

    #[allow(clippy::too_many_arguments)]
    fn yaz(
        &self,
        oyun: usize,
        toplam: usize,
        a_once: bool,
        skor: (usize, usize),
        fenler: &[String],
        hamleler: &[String],
        bitti: Option<&str>,
    ) {
        let liste = |v: &[String]| {
            v.iter()
                .map(|x| format!("\"{}\"", json_kac(x)))
                .collect::<Vec<_>>()
                .join(",")
        };
        let govde = format!(
            r#"{{{},"oyun":{oyun},"toplam":{toplam},"a_once":{a_once},"skor":[{},{}],"bitti":{},"fenler":[{}],"hamleler":[{}]}}"#,
            self.bas,
            skor.0,
            skor.1,
            bitti.map(|b| format!("\"{b}\"")).unwrap_or_else(|| "null".into()),
            liste(fenler),
            liste(hamleler)
        );
        // Önce geçici dosyaya yaz, sonra üstüne taşı: sunucu yarım yazılmış
        // bir dosya okumasın.
        let gecici = format!("{}.tmp", self.yol);
        if std::fs::write(&gecici, govde).is_ok() {
            let _ = std::fs::rename(&gecici, &self.yol);
        }
    }
}

fn json_kac(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// `izle` her hamleden sonra çağrılır: (o ana kadarki fenler, hamle adları).
fn play_from(
    start: &Position,
    a: &mut Engine,
    b: &mut Engine,
    a_is_first: bool,
    mut izle: impl FnMut(&[String], &[String]),
) -> Option<usize> {
    let mut pos = *start;
    let mut fenler = vec![to_fen(&pos)];
    let mut hamleler: Vec<String> = Vec::new();
    izle(&fenler, &hamleler);
    while pos.winner().is_none() && pos.ply < MAX_PLY {
        // side 0 her zaman alttan başlayan taraf.
        let a_turn = (pos.side == 0) == a_is_first;
        let e = if a_turn { &mut *a } else { &mut *b };
        let mv = e.choose(&pos).mv;
        let Some(mv) = mv else { return None };
        hamleler.push(move_name(&pos, mv));
        pos.make(mv);
        fenler.push(to_fen(&pos));
        izle(&fenler, &hamleler);
    }
    let w = pos.winner()?;
    // A kazandı mı?
    Some(usize::from((w == 0) != a_is_first))
}

fn cmd_match(args: &[String]) {
    let spec_a = arg(args, 2, "mcts:20000");
    let spec_b = arg(args, 3, "ab:1000ms");
    let pairs: usize = num(args, 4, 8);
    let seed: u64 = num(args, 5, 777);
    let open_plies: usize = num(args, 6, 4);

    let mut a_wins = 0usize;
    let mut b_wins = 0usize;
    let mut unfinished = 0usize;

    println!(
        "A = {spec_a}   B = {spec_b}\n{pairs} açılış x 2 renk = {} maç, açılış {open_plies} yarım hamle\n",
        pairs * 2
    );
    let t = Instant::now();
    let toplam_oyun = pairs * 2;
    let canli = Canli::yeni(&spec_a, &spec_b);
    // Oyun başına bir satır basıyoruz, açılış çifti başına değil. Sebebi
    // kalan süre tahmini: 2M iterasyonlu bir maçta bir açılış çifti 10+
    // dakika sürüyor, o çözünürlükte ETA hesaplanamıyor.
    for g in 0..pairs {
        let start = random_opening(seed + g as u64, open_plies);
        for (k, a_first) in [true, false].into_iter().enumerate() {
            let mut a = Engine::parse(&spec_a, seed + g as u64 * 7 + 1);
            let mut b = Engine::parse(&spec_b, seed + g as u64 * 13 + 2);
            let oyun_no = g * 2 + k + 1;
            let (sa, sb) = (a_wins, b_wins);
            let mut son: (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
            let sonuc_ham = play_from(&start, &mut a, &mut b, a_first, |f, h| {
                if let Some(c) = &canli {
                    c.yaz(oyun_no, toplam_oyun, a_first, (sa, sb), f, h, None);
                    son = (f.to_vec(), h.to_vec());
                }
            });
            let sonuc = match sonuc_ham {
                Some(0) => {
                    a_wins += 1;
                    "A"
                }
                Some(_) => {
                    b_wins += 1;
                    "B"
                }
                None => {
                    unfinished += 1;
                    "-"
                }
            };
            if let Some(c) = &canli {
                c.yaz(oyun_no, toplam_oyun, a_first, (a_wins, b_wins), &son.0, &son.1, Some(sonuc));
            }
            let oynanan = oyun_no;
            println!(
                "oyun {oynanan:>4}/{toplam_oyun}  açılış {:>3} {}  kazanan {sonuc}  {:.0} sn  [{a_wins}-{b_wins}]",
                g + 1,
                if a_first { "(A önce)" } else { "(B önce)" },
                t.elapsed().as_secs_f64()
            );
        }
    }
    let played = (a_wins + b_wins) as f64;
    let wr = if played > 0.0 {
        a_wins as f64 / played
    } else {
        0.0
    };
    println!(
        "\nA {a_wins} - {b_wins} B{}   A kazanma oranı {:.1}%   ({:.1} s)",
        if unfinished > 0 {
            format!("  ({unfinished} bitmedi)")
        } else {
            String::new()
        },
        wr * 100.0,
        t.elapsed().as_secs_f64()
    );
    if played >= 2.0 {
        // Normal yaklaşımla kaba %95 güven aralığı.
        let se = (wr * (1.0 - wr) / played).sqrt();
        println!(
            "kaba %95 aralık: {:.1}% – {:.1}%",
            (wr - 1.96 * se).max(0.0) * 100.0,
            (wr + 1.96 * se).min(1.0) * 100.0
        );
    }
}

fn cmd_play(args: &[String]) {
    let human: u8 = num(args, 2, 0);
    let spec = arg(args, 3, "mcts:2000ms");
    let mut engine = Engine::parse(&spec, 4242);
    let mut pos = Position::start();
    let stdin = io::stdin();

    println!(
        "Sen {} tarafısın. Rakip: {}. 'moves' hamleleri listeler, 'quit' çıkar.\n",
        if human == 0 {
            "A (alttan yukarı)"
        } else {
            "B (üstten aşağı)"
        },
        engine.name()
    );

    loop {
        if pos.side == human || pos.winner().is_some() {
            println!("{}", board_string(&pos));
        }
        if let Some(w) = pos.winner() {
            println!(
                ">>> {} kazandı.",
                if w == human as usize { "SEN" } else { "BOT" }
            );
            return;
        }
        if pos.ply >= MAX_PLY {
            println!(">>> Hamle sınırı.");
            return;
        }

        if pos.side == human {
            print!("hamlen> ");
            io::stdout().flush().ok();
            let mut line = String::new();
            if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
                return;
            }
            let line = line.trim();
            match line {
                "quit" | "q" => return,
                "moves" => {
                    let mut names: Vec<String> = pos
                        .legal_moves()
                        .iter()
                        .map(|&m| move_name(&pos, m))
                        .collect();
                    names.sort();
                    println!("{} legal hamle: {}\n", names.len(), names.join(" "));
                    continue;
                }
                _ => {}
            }
            match parse_move(&pos, line) {
                Some(mv) => {
                    pos.make(mv);
                }
                None => {
                    println!("Geçersiz hamle: '{line}'  ('moves' ile listeye bak)\n");
                    continue;
                }
            }
        } else {
            let c = engine.choose(&pos);
            let (mv, info, work) = (c.mv, c.info, c.work);
            let Some(mv) = mv else {
                println!("Bot hamle bulamadı.");
                return;
            };
            println!("bot: {}   ({info}, iş {work})\n", move_name(&pos, mv));
            pos.make(mv);
        }
    }
}

fn cmd_serve(args: &[String]) {
    let port: u16 = num(args, 2, 8080);
    if let Err(e) = gedik::server::serve(port) {
        eprintln!("sunucu başlatılamadı ({port}): {e}");
    }
}

fn cmd_tournament(args: &[String]) {
    let pairs: usize = num(args, 2, 4);
    let specs: Vec<String> = args.iter().skip(3).cloned().collect();
    if specs.len() < 2 {
        println!("kullanim: gedik tournament <acilis> <motor1> <motor2> [motor3...]");
        return;
    }
    let n = specs.len();
    let mut score = vec![0.0f64; n];
    let mut played = vec![0.0f64; n];
    let mut wins = vec![0u32; n];
    let mut losses = vec![0u32; n];
    let mut matrix = vec![vec![0.0f64; n]; n];
    let mut pairings: Vec<gedik::rating::Pairing> = Vec::new();

    // Kac thread? Motorlar tek thread kaliyor (t= verilmedikce); paralellik
    // oyunlara gidiyor. 16 cekirdekli makinede turnuva 30 kat hizli bitiyor
    // ve seri surumle ayni sonucu veriyor: her oyunun seed'i (i,j,g,renk)
    // dorduzunden turuyor, calisma sirasindan degil.
    let threads = std::env::var("TURNUVA_THREAD")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or_else(gedik::engine::default_threads);

    println!(
        "Turnuva: {} motor, cift basina {} acilis x 2 renk, {} thread\n",
        n, pairs, threads
    );
    let t = Instant::now();

    // Butun oyunlari once liste haline getiriyoruz, sonra is kuyrugundan
    // dagitiyoruz. Ciftler cok farkli hizlarda kosuyor (2M iterasyonluk bir
    // motor 20 binlikten 100 kat yavas), o yuzden cift basina degil oyun
    // basina dagitmak lazim; yoksa bir yavas cift butun makineyi bekletir.
    struct Gorev { i: usize, j: usize, g: usize, a_first: bool }
    let mut gorevler: Vec<Gorev> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            for g in 0..pairs {
                for a_first in [true, false] {
                    gorevler.push(Gorev { i, j, g, a_first });
                }
            }
        }
    }
    let toplam = gorevler.len();

    let sira = std::sync::atomic::AtomicUsize::new(0);
    let biten = std::sync::atomic::AtomicUsize::new(0);
    let sonuclar: Vec<std::sync::Mutex<Vec<(usize, usize, f64)>>> =
        (0..threads).map(|_| std::sync::Mutex::new(Vec::new())).collect();

    std::thread::scope(|scope| {
        for tid in 0..threads {
            let (gorevler, sira, biten, sonuclar, specs) =
                (&gorevler, &sira, &biten, &sonuclar, &specs);
            scope.spawn(move || loop {
                let k = sira.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(gv) = gorevler.get(k) else { break };
                let start = random_opening(
                    1000 + gv.g as u64 * 17 + (gv.i * 31 + gv.j) as u64,
                    4,
                );
                let mut ea = Engine::parse(&specs[gv.i], 7 + gv.g as u64 * 13 + gv.i as u64);
                let mut eb = Engine::parse(&specs[gv.j], 11 + gv.g as u64 * 19 + gv.j as u64);
                // i'nin bu oyundan aldigi puan: 1 kazanma, 0.5 bitmedi, 0 kayip
                let puan = match play_from(&start, &mut ea, &mut eb, gv.a_first, |_, _| {}) {
                    Some(0) => 1.0,
                    Some(_) => 0.0,
                    None => 0.5,
                };
                sonuclar[tid].lock().unwrap().push((gv.i, gv.j, puan));
                let b = biten.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if b % 20 == 0 || b == toplam {
                    println!("  {b}/{toplam} oyun  ({:.0} sn)", t.elapsed().as_secs_f64());
                }
            });
        }
    });

    // Toparlama: hangi thread'in hangi oyunu aldigi onemsiz, toplamlar ayni.
    let mut cift: std::collections::BTreeMap<(usize, usize), (f64, f64)> =
        std::collections::BTreeMap::new();
    for kova in &sonuclar {
        for &(i, j, puan) in kova.lock().unwrap().iter() {
            let e = cift.entry((i, j)).or_insert((0.0, 0.0));
            e.0 += puan;
            e.1 += 1.0;
            if puan > 0.5 {
                wins[i] += 1;
                losses[j] += 1;
            } else if puan < 0.5 {
                wins[j] += 1;
                losses[i] += 1;
            }
        }
    }
    println!();
    for (&(i, j), &(si, games)) in &cift {
        matrix[i][j] = si / games;
        matrix[j][i] = 1.0 - si / games;
        score[i] += si;
        score[j] += games - si;
        played[i] += games;
        played[j] += games;
        pairings.push(gedik::rating::Pairing { a: i, b: j, a_score: si, games });
        println!(
            "  {:<30} vs {:<30}  {:>4.1} - {:<4.1}",
            specs[i], specs[j], si, games - si
        );
    }

    let elo = gedik::rating::solve(n, &pairings, 4000);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| elo[b].total_cmp(&elo[a]));

    println!("\n{:<4}{:<24}{:>8}{:>8}{:>10}{:>9}", "#", "motor", "Elo", "+/-", "skor", "mac");
    for (rank, &i) in order.iter().enumerate() {
        let err = gedik::rating::elo_stderr(score[i], played[i]);
        println!(
            "{:<4}{:<24}{:>8.0}{:>8.0}{:>9.1}%{:>9.0}",
            rank + 1,
            specs[i],
            elo[i],
            1.96 * err,
            100.0 * score[i] / played[i].max(1.0),
            played[i]
        );
    }
    println!("\n{:.0} saniye", t.elapsed().as_secs_f64());

    // JSON: arayuz bunu okuyup guc tablosunu gosteriyor.
    let mut js = String::from("{\n  \"pairs\": ");
    js.push_str(&pairs.to_string());
    js.push_str(",\n  \"engines\": [\n");
    for (k, &i) in order.iter().enumerate() {
        let err = gedik::rating::elo_stderr(score[i], played[i]);
        js.push_str(&format!(
            "    {{\"rank\": {}, \"spec\": \"{}\", \"elo\": {:.0}, \"err\": {:.0}, \"score\": {:.4}, \"games\": {:.0}, \"wins\": {}, \"losses\": {}}}{}\n",
            k + 1,
            specs[i],
            elo[i],
            1.96 * err,
            score[i] / played[i].max(1.0),
            played[i],
            wins[i],
            losses[i],
            if k + 1 == n { "" } else { "," }
        ));
    }
    js.push_str("  ]\n}\n");
    if let Err(e) = std::fs::write("ratings.json", &js) {
        eprintln!("ratings.json yazilamadi: {e}");
    } else {
        println!("ratings.json yazildi");
    }
}

