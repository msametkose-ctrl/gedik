//! Sıfır bağımlılıklı yerel HTTP sunucusu + gömülü tek sayfa arayüz.
//!
//! API **durumsuz**: tarayıcı pozisyonu FEN dizisi olarak tutuyor, her istekte
//! gönderiyor. Sunucuda oturum yok — sayfayı yenilemek oyunu bozmuyor, iki
//! sekmede iki farklı oyun oynanabiliyor, ve `curl` ile motoru doğrudan
//! sorgulamak mümkün.
//!
//! | uç nokta | ne yapar |
//! |---|---|
//! | `GET /` | arayüz |
//! | `GET /api/state?fen=` | legal hamleler, duvarlar, mesafeler, kazanan |
//! | `GET /api/apply?fen=&move=<id>` | hamleyi uygular, yeni FEN döner |
//! | `GET /api/bot?fen=&engine=mcts:1000ms` | motorun hamlesi + arama bilgisi |
//! | `GET /api/replay?moves=e2,e8,e5h` | hamle listesinden pozisyon kurar |
//! | `GET /api/ratings` | turnuva sonuçları ve Elo tablosu |
//! | `GET /api/cores` | makinedeki çekirdek sayısı
//! | `GET /api/deney` | `deney/` klasöründeki maçların canlı toplamı |

use crate::bitboard::WSLOTS;
use crate::board::Position;
use crate::engine::Engine;
use crate::moves::{Move, MoveKind};
use crate::notation::{from_fen, move_name, to_fen};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Instant;

const UI: &str = include_str!("ui.html");
/// Derleme zamanında gömülen turnuva sonuçları. Çalışma dizininde
/// `ratings.json` varsa (yani kullanıcı kendi turnuvasını çalıştırdıysa) o
/// öncelikli.
const BUILTIN_RATINGS: &str = include_str!("ratings_builtin.json");
static SEED: AtomicU64 = AtomicU64::new(0x9e37_79b9_7f4a_7c15);

pub fn serve(port: u16) -> std::io::Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    println!("Quoridor arayüzü:  http://127.0.0.1:{port}");
    println!("Durdurmak için Ctrl-C.\n");
    for stream in listener.incoming() {
        if let Ok(s) = stream {
            thread::spawn(move || {
                let _ = handle(s);
            });
        }
    }
    Ok(())
}

fn handle(mut stream: TcpStream) -> std::io::Result<()> {
    let mut buf = [0u8; 8192];
    let n = stream.read(&mut buf)?;
    let req = String::from_utf8_lossy(&buf[..n]);
    let line = req.lines().next().unwrap_or("");
    let mut it = line.split_whitespace();
    let _method = it.next().unwrap_or("");
    let target = it.next().unwrap_or("/");
    let (path, query) = target.split_once('?').unwrap_or((target, ""));

    match path {
        "/" | "/index.html" => respond(&mut stream, 200, "text/html; charset=utf-8", UI),
        "/api/state" => api(&mut stream, query, api_state),
        "/api/apply" => api(&mut stream, query, api_apply),
        "/api/bot" => api(&mut stream, query, api_bot),
        "/api/replay" => api(&mut stream, query, api_replay),
        "/api/ratings" => {
            let body = std::fs::read_to_string("ratings.json")
                .unwrap_or_else(|_| BUILTIN_RATINGS.to_string());
            respond(&mut stream, 200, "application/json; charset=utf-8", &body)
        }
        "/api/deney" => {
            let body = deney_ozet();
            respond(&mut stream, 200, "application/json; charset=utf-8", &body)
        }
        "/api/cores" => {
            let body = format!("{{\"cores\": {}}}", crate::engine::default_threads());
            respond(&mut stream, 200, "application/json; charset=utf-8", &body)
        }
        _ => respond(&mut stream, 404, "text/plain; charset=utf-8", "yok"),
    }
}

fn api<F>(stream: &mut TcpStream, query: &str, f: F) -> std::io::Result<()>
where
    F: Fn(&str) -> Result<String, String>,
{
    match f(query) {
        Ok(body) => respond(stream, 200, "application/json; charset=utf-8", &body),
        Err(e) => respond(
            stream,
            400,
            "application/json; charset=utf-8",
            &format!("{{\"error\":\"{}\"}}", escape(&e)),
        ),
    }
}

fn respond(
    stream: &mut TcpStream,
    status: u16,
    ctype: &str,
    body: &str,
) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        _ => "Not Found",
    };
    let head = format!(
        // Sunucu yalnızca 127.0.0.1'e bağlı. CORS açık, böylece motoru
        // başka bir yerel sayfadan ya da tarayıcı konsolundan `fetch` ile
        // denemek mümkün.
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()
}

/// `deney/` klasöründeki maç çıktılarını okuyup deney başına toplar.
///
/// `deney.bat` her deneyi farklı seed'lerle birden fazla sürece bölüyor
/// (`3-yumusak-deger__1.txt`, `__2.txt`, ...). Buradaki iş dosya adındaki
/// `__parça` ekini atıp aynı deneyin parçalarını toplamak. Biten parçadan
/// `A n - m B` satırını, devam eden parçadan son `[n-m]` ara skorunu
/// alıyoruz — böylece tablo maç sürerken de anlamlı.
fn deney_ozet() -> String {
    let mut gruplar: Vec<(String, u32, u32, u32, u32)> = Vec::new();

    let Ok(dir) = std::fs::read_dir("deney") else {
        return "{\"deneyler\":[]}".to_string();
    };
    let mut dosyalar: Vec<std::path::PathBuf> =
        dir.filter_map(|e| e.ok().map(|e| e.path())).collect();
    dosyalar.sort();

    for path in dosyalar {
        if path.extension().and_then(|e| e.to_str()) != Some("txt") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { continue };
        let ad = stem.split("__").next().unwrap_or(stem).to_string();
        let metin = std::fs::read_to_string(&path).unwrap_or_default();

        // Biten parça mı?
        let bitti = metin
            .lines()
            .rev()
            .find_map(|l| l.strip_prefix("A ").and_then(skor_ciz));
        // Devam eden parçanın son ara skoru: satır sonundaki [n-m]
        let ara = metin.lines().rev().find_map(|l| {
            let i = l.rfind('[')?;
            let j = l.rfind(']')?;
            if j < i { return None }
            let (a, b) = l[i + 1..j].split_once('-')?;
            Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
        });

        let (a, b, tam) = match bitti {
            Some((a, b)) => (a, b, 1),
            None => match ara {
                Some((a, b)) => (a, b, 0),
                None => (0, 0, 0),
            },
        };

        match gruplar.iter_mut().find(|g| g.0 == ad) {
            Some(g) => {
                g.1 += a;
                g.2 += b;
                g.3 += tam;
                g.4 += 1;
            }
            None => gruplar.push((ad, a, b, tam, 1)),
        }
    }

    let items: Vec<String> = gruplar
        .iter()
        .map(|(ad, a, b, tam, parca)| {
            format!(
                r#"{{"ad":"{}","a":{},"b":{},"biten":{},"parca":{}}}"#,
                escape(ad), a, b, tam, parca
            )
        })
        .collect();
    format!("{{\"deneyler\":[{}]}}", items.join(","))
}

/// `"17 - 3 B"` -> `(17, 3)`. Maç özetinin ilk satırı `A 17 - 3 B ...`.
fn skor_ciz(rest: &str) -> Option<(u32, u32)> {
    let (a, kalan) = rest.split_once(" - ")?;
    let b = kalan.split_whitespace().next()?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

// ------------------------------------------------------------------ yardımcı

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Query parametresi, yüzde ve `+` çözümlemesiyle.
fn param(query: &str, key: &str) -> Option<String> {
    for pair in query.split('&') {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        if k != key {
            continue;
        }
        let bytes = v.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            match bytes[i] {
                b'+' => {
                    out.push(b' ');
                    i += 1;
                }
                b'%' if i + 2 < bytes.len() => {
                    let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok()?;
                    out.push(u8::from_str_radix(hex, 16).ok()?);
                    i += 3;
                }
                c => {
                    out.push(c);
                    i += 1;
                }
            }
        }
        return String::from_utf8(out).ok();
    }
    None
}

fn position_param(query: &str) -> Result<Position, String> {
    let fen = param(query, "fen").unwrap_or_else(|| to_fen(&Position::start()));
    from_fen(&fen).ok_or_else(|| format!("geçersiz fen: {fen}"))
}

fn move_json(pos: &Position, mv: Move) -> String {
    let name = move_name(pos, mv);
    match mv.kind() {
        MoveKind::Pawn(d) => format!(
            r#"{{"id":{},"kind":"p","to":{},"name":"{}"}}"#,
            mv.action_id(),
            pos.pawn_target(d),
            name
        ),
        MoveKind::HWall(s) => format!(
            r#"{{"id":{},"kind":"h","slot":{},"name":"{}"}}"#,
            mv.action_id(),
            s,
            name
        ),
        MoveKind::VWall(s) => format!(
            r#"{{"id":{},"kind":"v","slot":{},"name":"{}"}}"#,
            mv.action_id(),
            s,
            name
        ),
    }
}

fn slots_json(bits: u64) -> String {
    let v: Vec<String> = (0..WSLOTS)
        .filter(|&s| bits >> s & 1 != 0)
        .map(|s| s.to_string())
        .collect();
    format!("[{}]", v.join(","))
}

fn opt_num(x: Option<u32>) -> String {
    x.map(|v| v.to_string()).unwrap_or_else(|| "null".into())
}

// -------------------------------------------------------------- uç noktalar

fn api_state(query: &str) -> Result<String, String> {
    let pos = position_param(query)?;
    let moves: Vec<String> = pos
        .legal_moves()
        .iter()
        .map(|&mv| move_json(&pos, mv))
        .collect();
    Ok(format!(
        r#"{{"fen":"{}","side":{},"ply":{},"winner":{},"pawn":[{},{}],"walls":[{},{}],"dist":[{},{}],"h":{},"v":{},"moves":[{}]}}"#,
        to_fen(&pos),
        pos.side,
        pos.ply,
        pos.winner()
            .map(|w| w.to_string())
            .unwrap_or_else(|| "null".into()),
        pos.pawn[0],
        pos.pawn[1],
        pos.walls[0],
        pos.walls[1],
        opt_num(pos.distance(0)),
        opt_num(pos.distance(1)),
        slots_json(pos.h),
        slots_json(pos.v),
        moves.join(",")
    ))
}

fn api_apply(query: &str) -> Result<String, String> {
    let mut pos = position_param(query)?;
    let id: u16 = param(query, "move")
        .and_then(|s| s.parse().ok())
        .ok_or("move parametresi yok")?;
    if id as usize >= crate::moves::NUM_ACTIONS {
        return Err(format!("aksiyon id aralık dışı: {id}"));
    }
    let mv = Move(id);
    if !pos.is_legal(mv) {
        return Err(format!("illegal hamle: {}", move_name(&pos, mv)));
    }
    let name = move_name(&pos, mv);
    pos.make(mv);
    Ok(format!(
        r#"{{"fen":"{}","played":"{}"}}"#,
        to_fen(&pos),
        escape(&name)
    ))
}

fn api_bot(query: &str) -> Result<String, String> {
    let pos = position_param(query)?;
    if pos.winner().is_some() {
        return Err("oyun bitmiş".into());
    }
    let spec = param(query, "engine").unwrap_or_else(|| "mcts:1000ms".into());
    let seed = SEED.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed) | 1;
    let mut engine = Engine::parse(&spec, seed);

    let t = Instant::now();
    let choice = engine.choose(&pos);
    let elapsed = t.elapsed().as_secs_f64();

    let mv = choice.mv.ok_or("motor hamle bulamadı")?;
    let top: Vec<String> = choice
        .top
        .iter()
        .take(6)
        .map(|(m, visits, wr)| {
            format!(
                r#"{{"name":"{}","visits":{},"wr":{:.4}}}"#,
                escape(&move_name(&pos, *m)),
                visits,
                wr
            )
        })
        .collect();

    let mut after = pos;
    after.make(mv);
    Ok(format!(
        r#"{{"id":{},"name":"{}","engine":"{}","info":"{}","work":{},"elapsed":{:.3},"top":[{}],"fen":"{}"}}"#,
        mv.action_id(),
        escape(&move_name(&pos, mv)),
        escape(&engine.name()),
        escape(&choice.info),
        choice.work,
        elapsed,
        top.join(","),
        to_fen(&after)
    ))
}

/// Başlangıç pozisyonundan bir hamle listesini oynatır.
///
/// Tahtayı piksel piksel okumak yerine hamle listesini yeniden oynatmak çok
/// daha sağlam: sitenin çizim biçimi değişse bile notasyon değişmez, ve
/// pozisyonu kuran taraf kuralları zaten bilen motor oluyor.
///
/// `moves` virgül ya da boşlukla ayrılmış Glendenning notasyonu:
/// `e2,e8,e5h,g7v`. İlk illegal hamlede durur ve nerede takıldığını söyler.
fn api_replay(query: &str) -> Result<String, String> {
    let raw = param(query, "moves").unwrap_or_default();
    let mut pos = Position::start();
    let mut applied = 0usize;
    let mut names: Vec<String> = Vec::new();

    for tok in raw.split([',', ' ', '\n']).filter(|t| !t.trim().is_empty()) {
        let tok = tok.trim();
        if pos.winner().is_some() {
            return Err(format!("oyun {applied}. hamlede bitmişti, fazladan hamle: {tok}"));
        }
        let mv = crate::notation::parse_move(&pos, tok)
            .ok_or_else(|| format!("{}. hamle okunamadı ya da illegal: '{tok}'", applied + 1))?;
        names.push(move_name(&pos, mv));
        pos.make(mv);
        applied += 1;
    }

    Ok(format!(
        r#"{{"fen":"{}","ply":{},"side":{},"applied":{},"winner":{},"moves":[{}]}}"#,
        to_fen(&pos),
        pos.ply,
        pos.side,
        applied,
        pos.winner()
            .map(|w| w.to_string())
            .unwrap_or_else(|| "null".into()),
        names
            .iter()
            .map(|n| format!("\"{}\"", escape(n)))
            .collect::<Vec<_>>()
            .join(",")
    ))
}
