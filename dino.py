import os
import copy
import time
import math
import queue
import signal
import pickle
import itertools
import multiprocessing as mp

import neat
from playwright.sync_api import sync_playwright

GAME_URL = "https://wayou.github.io/t-rex-runner/"

# ---------- НАСТРОЙКИ ПАРАЛЛЕЛЬНОСТИ ----------
NUM_WORKERS = 7
HEADLESS = True

# ---------- НАСТРОЙКИ NEAT И СОХРАНЕНИЯ ----------
CONFIG_FILE = "config-neat.txt"
MODEL_FILE = "best_dino_v2.pkl"
CHECKPOINT_DIR = "checkpoints_v2"
CHECKPOINT_PREFIX = "neat-checkpoint-"
KEEP_CHECKPOINTS = 5

TRAIN_MODE = False
RESUME_TRAINING = True
GENERATIONS = 20
TEST_GAMES = 20

# ---------- НАСТРОЙКИ УСКОРЕНИЯ ----------
TRAIN_SPEEDUP = 2
MAX_TRAIN_SCORE = 3000.0
TRAIN_START_SPEEDS = [6.0, 11.0]
POLL_INTERVAL = 0.08
STALL_TIMEOUT = 8.0
TRAIN_MAX_SECONDS = 120.0

ANTI_PAUSE_JS = """
() => {
    const blocked = new Set(['blur', 'visibilitychange', 'webkitvisibilitychange']);
    const origWinAdd = window.addEventListener.bind(window);
    window.addEventListener = function(type, listener, options) {
        if (blocked.has(type)) return;
        return origWinAdd(type, listener, options);
    };
    const origDocAdd = document.addEventListener.bind(document);
    document.addEventListener = function(type, listener, options) {
        if (blocked.has(type)) return;
        return origDocAdd(type, listener, options);
    };
    Object.defineProperty(document, 'hidden', { get: () => false });
    Object.defineProperty(document, 'webkitHidden', { get: () => false });
    Object.defineProperty(document, 'visibilityState', { get: () => 'visible' });
}
"""

# ИСПРАВЛЕНО: __neatStopRun теперь ВНУТРИ, до return true
BOT_ENGINE_JS = """
() => {
    if (window.__neatEngineInstalled) return true;
    const runner = window.Runner && window.Runner.instance_;
    if (!runner) return false;

    window.__neatEngineInstalled = true;
    window.__neatRun = null;

    const JUMP_ON = 0.55, JUMP_OFF = 0.35;
    const DUCK_ON = 0.55, DUCK_OFF = 0.35;
    const STICK_MARGIN = 0.15;
    const JUMP_COMMIT_FRAMES = 9;
    const DUCK_MIN_HOLD = 5;

    function activateFn(name, z) {
        switch (name) {
            case 'tanh': return Math.tanh(Math.max(-60.0, Math.min(60.0, 2.5 * z)));
            case 'sigmoid': return 1.0 / (1.0 + Math.exp(-Math.max(-60.0, Math.min(60.0, 5.0 * z))));
            case 'relu': return z > 0.0 ? z : 0.0;
            case 'identity': return z;
            case 'clamped': return Math.max(-1.0, Math.min(1.0, z));
            case 'sin': return Math.sin(Math.max(-60.0, Math.min(60.0, 5.0 * z)));
            case 'abs': return Math.abs(z);
            case 'square': return z * z;
            case 'cube': return z * z * z;
            case 'exp': return Math.exp(Math.max(-60.0, Math.min(60.0, z)));
            case 'gauss': {
                const c = Math.max(-3.4, Math.min(3.4, z));
                return Math.exp(-5.0 * c * c);
            }
            case 'hat': return Math.max(0.0, 1.0 - Math.abs(z));
            case 'inv': return z === 0.0 ? 0.0 : 1.0 / z;
            case 'log': return Math.log(Math.max(1e-7, z));
            case 'softplus': {
                const c = Math.max(-60.0, Math.min(60.0, 5.0 * z));
                return 0.2 * Math.log(1.0 + Math.exp(c));
            }
            case 'elu': return z > 0.0 ? z : Math.exp(z) - 1.0;
            case 'lelu': return z > 0.0 ? z : 0.005 * z;
            default: return Math.tanh(Math.max(-60.0, Math.min(60.0, 2.5 * z)));
        }
    }

    function evalNetwork(net, inputs) {
        const vals = Object.create(null);
        for (let i = 0; i < net.inputs.length; i++) vals[net.inputs[i]] = inputs[i];
        for (let i = 0; i < net.outputs.length; i++) vals[net.outputs[i]] = 0.0;
        const evals = net.evals;
        for (let i = 0; i < evals.length; i++) {
            const ev = evals[i];
            const links = ev.links;
            let s = 0.0;
            if (ev.agg === 'sum') {
                for (let j = 0; j < links.length; j++) s += (vals[links[j][0]] || 0.0) * links[j][1];
            } else if (ev.agg === 'product') {
                s = 1.0;
                for (let j = 0; j < links.length; j++) s *= (vals[links[j][0]] || 0.0) * links[j][1];
            } else if (ev.agg === 'max') {
                s = links.length ? -Infinity : 0.0;
                for (let j = 0; j < links.length; j++) {
                    const v = (vals[links[j][0]] || 0.0) * links[j][1];
                    if (v > s) s = v;
                }
            } else if (ev.agg === 'min') {
                s = links.length ? Infinity : 0.0;
                for (let j = 0; j < links.length; j++) {
                    const v = (vals[links[j][0]] || 0.0) * links[j][1];
                    if (v < s) s = v;
                }
            } else if (ev.agg === 'mean') {
                for (let j = 0; j < links.length; j++) s += (vals[links[j][0]] || 0.0) * links[j][1];
                if (links.length) s /= links.length;
            }
            vals[ev.node] = activateFn(ev.act, ev.bias + ev.response * s);
        }
        const out = new Array(net.outputs.length);
        for (let i = 0; i < net.outputs.length; i++) out[i] = vals[net.outputs[i]] || 0.0;
        return out;
    }

    function sanitizeTrex(r) {
        const t = r.tRex;
        if (!t || r.crashed || !r.playing) return;
        try {
            if (t.jumping && t.ducking) t.setDuck(false);
            if (!t.jumping && t.speedDrop) t.speedDrop = false;
            if (!t.jumping) {
                if (t.ducking && t.status !== 'DUCKING') t.update(0, 'DUCKING');
                else if (!t.ducking && t.status === 'DUCKING') t.update(0, 'RUNNING');
            }
        } catch(e){}
    }

    function releaseAllKeys(r, run) {
        if (run) {
            run.jumpHeld = false; run.duckHeld = false;
            run.wantJump = false; run.wantDuck = false;
            run.decided = 'run'; run.jumpCommit = 0; run.duckHold = 0;
        }
        if (!r || !r.tRex) return;
        const t = r.tRex;
        try { if (t.jumping) t.endJump(); } catch(e){}
        try { t.speedDrop = false; } catch(e){}
        try { if (t.ducking) t.setDuck(false); } catch(e){ try { t.ducking = false; } catch(_){} }
    }

    function getInputs(r) {
        const dino = r.tRex;
        const obstacles = (r.horizon && r.horizon.obstacles) ? r.horizon.obstacles : [];
        const speed = r.currentSpeed || 6.0;
        const upcoming = [];
        for (let i = 0; i < obstacles.length; i++) {
            const obs = obstacles[i];
            const w = obs.width || (obs.typeConfig ? obs.typeConfig.width * (obs.size || 1) : 25);
            if (obs.xPos + w > dino.xPos + 4) {
                const h = obs.height || (obs.typeConfig ? obs.typeConfig.height : 40);
                upcoming.push({ x: obs.xPos, w: w, y: obs.yPos || 100, h: h });
                if (upcoming.length === 2) break;
            }
        }
        let dist1 = 600.0, obs1W = 0.0, obs1Y = 0.0, obs1H = 0.0, gap12 = 600.0;
        if (upcoming.length >= 1) {
            dist1 = upcoming[0].x - dino.xPos;
            obs1W = upcoming[0].w; obs1Y = upcoming[0].y; obs1H = upcoming[0].h;
        }
        if (upcoming.length >= 2) {
            gap12 = Math.max(0.0, upcoming[1].x - (upcoming[0].x + upcoming[0].w));
        }
        const groundY = dino.groundYPos || 93;
        const dinoAlt = Math.max(0.0, groundY - dino.yPos) / 100.0;
        return [dist1/600.0, obs1W/100.0, obs1Y/150.0, obs1H/60.0, gap12/600.0, speed/15.0, dinoAlt];
    }

    function stepBot(r, run) {
        sanitizeTrex(r);
        const tRex = r.tRex;
        if (!tRex || r.crashed || !r.playing) return;

        const inputs = getInputs(r);
        const out = evalNetwork(run.net, inputs);
        const jumpSignal = out[0] || 0.0;
        const duckSignal = out.length > 1 ? out[1] : -1.0;

        if (jumpSignal > JUMP_ON) run.wantJump = true;
        else if (jumpSignal < JUMP_OFF) run.wantJump = false;

        if (duckSignal > DUCK_ON) run.wantDuck = true;
        else if (duckSignal < DUCK_OFF) run.wantDuck = false;

        let wantJump = run.wantJump;
        let wantDuck = run.wantDuck;

        if (wantJump && wantDuck) {
            if (run.decided === 'jump' && jumpSignal + STICK_MARGIN >= duckSignal) wantDuck = false;
            else if (run.decided === 'duck' && duckSignal + STICK_MARGIN >= jumpSignal) wantJump = false;
            else if (jumpSignal >= duckSignal) wantDuck = false;
            else wantJump = false;
        }

        if (run.jumpCommit > 0) run.jumpCommit--;
        if (run.duckHold > 0) run.duckHold--;

        if (run.jumpCommit > 0 && wantDuck && !wantJump) {
            if (duckSignal < 0.85) wantDuck = false;
        }
        if (run.duckHold > 0 && !wantJump && !wantDuck && !tRex.jumping) {
            wantDuck = true;
        }

        if (wantJump && !wantDuck) {
            run.decided = 'jump';
            if (tRex.ducking) {
                try { tRex.setDuck(false); } catch(e){}
                run.duckHeld = false; run.duckHold = 0;
                return;
            }
            if (tRex.speedDrop) { try { tRex.speedDrop = false; } catch(e){} }
            if (!tRex.jumping) {
                try { tRex.startJump(r.currentSpeed); } catch(e){}
                run.jumpHeld = true;
                run.jumpCommit = JUMP_COMMIT_FRAMES;
            } else {
                run.jumpHeld = true;
            }
            run.duckHeld = false;
        } else if (wantDuck && !wantJump) {
            run.decided = 'duck';
            if (run.jumpHeld) { try { tRex.endJump(); } catch(e){} run.jumpHeld = false; }
            if (tRex.jumping) {
                if (!tRex.speedDrop) { try { tRex.setSpeedDrop(); } catch(e){} }
                run.duckHeld = true;
            } else {
                if (!tRex.ducking) {
                    try { tRex.setDuck(true); } catch(e){}
                    run.duckHold = DUCK_MIN_HOLD;
                }
                run.duckHeld = true;
            }
        } else {
            run.decided = 'run';
            if (run.jumpHeld) { try { tRex.endJump(); } catch(e){} run.jumpHeld = false; }
            if (tRex.speedDrop) { try { tRex.speedDrop = false; } catch(e){} }
            if (!tRex.jumping && tRex.ducking && run.duckHold <= 0) {
                try { tRex.setDuck(false); } catch(e){}
            }
            if (run.duckHold <= 0) run.duckHeld = false;
        }

        if (tRex.jumping && tRex.ducking) {
            try { tRex.setDuck(false); } catch(e){ tRex.ducking = false; }
        }
    }

    function finishRun(r, run) {
        try { releaseAllKeys(r, run); } catch(e){}
        run.active = false; run.done = true;
        try { r.stop(); r.paused = false; } catch(e){}
    }

    const origUpdate = runner.update.bind(runner);
    runner.update = function() {
        const run = window.__neatRun;
        if (!run || !run.active) return origUpdate();
        const steps = run.speedup || 1;
        const savedSchedule = runner.scheduleNextUpdate;
        runner.scheduleNextUpdate = function() {};
        try {
            for (let s = 0; s < steps; s++) {
                if (!runner.playing || runner.crashed) break;
                if (run.frame === 0 && run.startSpeed > 0) runner.currentSpeed = run.startSpeed;
                stepBot(runner, run);
                run.frame++;
                const now = performance.now();
                if (steps > 1 || !runner.time || (now - runner.time) > 25) {
                    runner.time = now - 16.666667;
                }
                origUpdate();
                const score = (runner.distanceRan || 0) * 0.025;
                if (score > run.score) run.score = score;
                if (runner.crashed || (run.maxScore > 0 && run.score >= run.maxScore)) {
                    finishRun(runner, run);
                    return;
                }
            }
        } catch (err) {
            run.error = String(err);
            finishRun(runner, run);
            return;
        } finally {
            runner.scheduleNextUpdate = savedSchedule;
        }
        if (runner.playing && !runner.crashed && run.active) {
            runner.updatePending = false;
            runner.scheduleNextUpdate();
        }
    };

    window.__neatStartRun = function(cfg) {
        if (runner.raqId) { cancelAnimationFrame(runner.raqId); runner.raqId = 0; }
        try {
            runner.tRex.speedDrop = false;
            if (runner.tRex.ducking) runner.tRex.setDuck(false);
        } catch(e){}
        runner.playing = false; runner.paused = false;
        runner.crashed = false; runner.updatePending = false;
        window.__neatRun = {
            net: cfg.net, speedup: cfg.speedup || 1,
            maxScore: cfg.maxScore || 0, startSpeed: cfg.startSpeed || 0,
            active: true, done: false, score: 0.0, frame: 0,
            jumpHeld: false, duckHeld: false,
            wantJump: false, wantDuck: false, decided: 'run',
            jumpCommit: 0, duckHold: 0, error: null
        };
        runner.restart();
        return true;
    };

    window.__neatGetStatus = function() {
        const run = window.__neatRun;
        if (!run) return { done: true, score: 0.0, frame: 0, error: 'NO_RUN' };
        return { done: !!run.done, score: run.score, frame: run.frame, error: run.error };
    };

    window.__neatStopRun = function() {
        const r = window.Runner && window.Runner.instance_;
        const run = window.__neatRun;
        if (!run) return;
        try { releaseAllKeys(r, run); } catch(e){}
        run.active = false; run.done = true;
        try { r.stop(); r.paused = false; } catch(e){}
    };

    return true;
}
"""


def save_genome(genome, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(genome, f)
    print(f"--> Новый рекорд! Геном сохранён в {filepath}", flush=True)


def serialize_network(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    node_evals = []
    for node, act_func, agg_func, bias, response, links in net.node_evals:
        ng = genome.nodes.get(node)
        act_name = ng.activation if ng is not None else getattr(act_func, "__name__", "tanh").replace("_activation", "")
        agg_name = ng.aggregation if ng is not None else getattr(agg_func, "__name__", "sum").replace("_aggregation", "")
        node_evals.append({
            "node": int(node),
            "act": str(act_name),
            "agg": str(agg_name),
            "bias": float(bias),
            "response": float(response),
            "links": [[int(i), float(w)] for i, w in links],
        })
    return {
        "inputs": [int(i) for i in net.input_nodes],
        "outputs": [int(i) for i in net.output_nodes],
        "evals": node_evals,
    }


def find_latest_checkpoint(checkpoint_dir, prefix=CHECKPOINT_PREFIX):
    if not os.path.isdir(checkpoint_dir):
        return None
    checkpoints = []
    for name in os.listdir(checkpoint_dir):
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if suffix.isdigit():
                checkpoints.append((int(suffix), os.path.join(checkpoint_dir, name)))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def clear_checkpoints(checkpoint_dir, prefix=CHECKPOINT_PREFIX):
    if not os.path.isdir(checkpoint_dir):
        return
    for name in os.listdir(checkpoint_dir):
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            try:
                os.remove(os.path.join(checkpoint_dir, name))
            except OSError:
                pass


class RollingCheckpointer(neat.Checkpointer):
    def __init__(self, generation_interval, filename_prefix, keep_last=5):
        super().__init__(
            generation_interval=generation_interval,
            time_interval_seconds=None,
            filename_prefix=filename_prefix,
        )
        self.keep_last = keep_last

    def save_checkpoint(self, config, population, species_set, generation):
        super().save_checkpoint(config, population, species_set, generation)
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        if not self.keep_last or self.keep_last <= 0:
            return
        folder = os.path.dirname(self.filename_prefix) or "."
        base_prefix = os.path.basename(self.filename_prefix)
        files = []
        for name in os.listdir(folder):
            if name.startswith(base_prefix):
                suffix = name[len(base_prefix):]
                if suffix.isdigit():
                    files.append((int(suffix), os.path.join(folder, name)))
        files.sort(key=lambda item: item[0])
        for _, old_path in files[:-self.keep_last]:
            try:
                os.remove(old_path)
            except OSError:
                pass


def is_genome_compatible(genome, config):
    expected_outputs = set(config.genome_config.output_keys)
    expected_inputs = set(config.genome_config.input_keys)
    if not expected_outputs.issubset(set(genome.nodes.keys())):
        return False
    for in_key, _ in genome.connections.keys():
        if in_key < 0 and in_key not in expected_inputs:
            return False
    return True


def seed_population_from_genome(pop, config, best_genome):
    genome_cfg = config.genome_config
    max_node_id = max(best_genome.nodes.keys(), default=0)
    min_next_node = max(max_node_id + 1, genome_cfg.num_outputs)
    genome_cfg.node_indexer = itertools.count(min_next_node)
    tracker = getattr(genome_cfg, "innovation_tracker", None)
    if tracker is not None and best_genome.connections:
        max_innov = max(
            (getattr(c, "innovation", 0) or 0 for c in best_genome.connections.values()),
            default=0,
        )
        next_innov = getattr(tracker, "next_innovation", None)
        if isinstance(next_innov, int) and next_innov <= max_innov:
            tracker.next_innovation = max_innov + 1
    num_exact = min(2, len(pop.population))
    new_population = {}
    for idx, gid in enumerate(list(pop.population.keys())):
        g = copy.deepcopy(best_genome)
        g.key = gid
        g.fitness = None
        if idx >= num_exact:
            g.mutate(genome_cfg)
            if idx % 3 == 0:
                g.mutate(genome_cfg)
        new_population[gid] = g
    pop.population = new_population
    pop.species = config.species_set_type(config.species_set_config, pop.reporters)
    pop.species.speciate(config, pop.population, pop.generation)


def bootstrap_game(page):
    deadline = time.time() + 10.0
    while time.time() < deadline:
        ready = page.evaluate("() => !!(window.Runner && window.Runner.instance_)")
        if ready:
            break
        time.sleep(0.1)
    canvas = page.locator("canvas")
    canvas.click()
    time.sleep(0.2)
    page.keyboard.press("Space")
    deadline = time.time() + 5.0
    while time.time() < deadline:
        state = page.evaluate(
            """() => {
                const r = window.Runner && window.Runner.instance_;
                if (!r) return null;
                return { activated: !!r.activated, playing: !!r.playing, intro: !!r.playingIntro, crashed: !!r.crashed };
            }"""
        )
        if state and ((state["playing"] and not state["intro"]) or state["crashed"]):
            break
        time.sleep(0.05)
    page.evaluate(
        """() => {
            const r = window.Runner && window.Runner.instance_;
            if (r) { r.stop(); r.paused = false; r.crashed = true; }
        }"""
    )
    ok = page.evaluate(BOT_ENGINE_JS)
    if not ok:
        raise RuntimeError("Не удалось установить JS-движок бота.")


def play_one_game(page, net_data, speedup=1, max_score=0.0,
                  start_speed=6.0, stop_event=None, max_seconds=None):
    start_time = time.time()
    for attempt in range(2):
        page.evaluate(
            "(cfg) => window.__neatStartRun(cfg)",
            {
                "net": net_data,
                "speedup": int(speedup),
                "maxScore": float(max_score),
                "startSpeed": float(start_speed),
            },
        )
        last_frame = -1
        last_progress = time.time()
        last_heartbeat = time.time()
        while True:
            if stop_event and stop_event.is_set():
                return 0.0
            if max_seconds is not None and time.time() - start_time > max_seconds:
                try:
                    page.evaluate("() => window.__neatStopRun()")
                    status = page.evaluate("() => window.__neatGetStatus()")
                    return float(status.get("score", 0.0))
                except Exception:
                    return 0.0
            time.sleep(POLL_INTERVAL)
            status = page.evaluate("() => window.__neatGetStatus()")
            if status["done"]:
                if status.get("error"):
                    print(f"Ошибка JS-движка: {status['error']}", flush=True)
                return float(status["score"])
            frame = status["frame"]
            if frame != last_frame:
                last_frame = frame
                last_progress = time.time()
            else:
                if time.time() - last_progress > STALL_TIMEOUT:
                    if attempt == 0:
                        page.reload(wait_until="networkidle", timeout=60000)
                        time.sleep(1.5)
                        bootstrap_game(page)
                        break
                    return float(status.get("score", 0.0))
            if time.time() - last_heartbeat > 5.0:
                print(f"   ... игра идёт: кадр={frame}, очки={status['score']:.0f}", flush=True)
                last_heartbeat = time.time()
    return 0.0


def create_browser_page(p, headless=False, worker_id=1, total_workers=1):
    launch_args = [
        "--disable-gpu",
        "--disable-extensions",
        "--mute-audio",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
    ]
    if not headless and total_workers > 1:
        cols = max(1, math.ceil(math.sqrt(total_workers)))
        win_w, win_h = 640, 320
        col = (worker_id - 1) % cols
        row = (worker_id - 1) // cols
        launch_args.append(f"--window-size={win_w},{win_h}")
        launch_args.append(f"--window-position={col * win_w},{row * win_h}")
    browser = p.chromium.launch(channel="chromium", headless=headless, args=launch_args)
    viewport = {"width": 620, "height": 260} if total_workers > 1 else {"width": 800, "height": 600}
    page = browser.new_page(viewport=viewport)
    page.add_init_script(ANTI_PAUSE_JS)
    page.goto(GAME_URL, wait_until="networkidle", timeout=60000)
    time.sleep(1.0)
    bootstrap_game(page)
    return browser, page


def worker_loop(worker_id, total_workers, config_path, task_queue, result_queue, stop_event, headless):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    try:
        with sync_playwright() as p:
            browser, page = create_browser_page(
                p, headless=headless, worker_id=worker_id, total_workers=total_workers,
            )
            print(f"[Воркер #{worker_id}] Браузер готов к работе.", flush=True)
            try:
                while not stop_event.is_set():
                    try:
                        task = task_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if task is None:
                        break
                    genome_id, genome = task
                    net_data = serialize_network(genome, config)
                    scores = []
                    for start_speed in TRAIN_START_SPEEDS:
                        if stop_event.is_set():
                            break
                        try:
                            # ИСПРАВЛЕНО: правильные train-параметры и переменная sc
                            sc = play_one_game(
                                page,
                                net_data,
                                speedup=TRAIN_SPEEDUP,
                                max_score=MAX_TRAIN_SCORE,
                                start_speed=start_speed,
                                stop_event=stop_event,
                                max_seconds=TRAIN_MAX_SECONDS,
                            )
                        except Exception as e:
                            print(f"[Воркер #{worker_id}] Ошибка в игре (ID={genome_id}): {e}", flush=True)
                            sc = 0.0
                        scores.append(sc)
                    if not stop_event.is_set() and scores:
                        avg_score = sum(scores) / len(scores)
                        min_score = min(scores)
                        fitness = 0.4 * min_score + 0.6 * avg_score if len(scores) > 1 else scores[0]
                        result_queue.put((worker_id, genome_id, fitness, scores))
            finally:
                try:
                    browser.close()
                except Exception:
                    pass
    except Exception as e:
        print(f"[Воркер #{worker_id}] Критическая ошибка запуска: {e}", flush=True)
    finally:
        task_queue.cancel_join_thread()
        result_queue.cancel_join_thread()


class ParallelDinoEvaluator:
    def __init__(self, num_workers, config_path, model_path, headless=False, start_generation=0):
        self.num_workers = num_workers
        self.config_path = config_path
        self.model_path = model_path
        self.headless = headless
        self.ctx = mp.get_context("spawn")
        self.task_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        self.stop_event = self.ctx.Event()
        self.workers = []
        self.generation_num = start_generation
        self.best_overall_fitness = -1.0
        self.best_overall_genome = None

    def start(self):
        print(f"Запускаем {self.num_workers} параллельных браузеров (headless={self.headless}, speedup={TRAIN_SPEEDUP}x, скорости={TRAIN_START_SPEEDS})...", flush=True)
        for i in range(1, self.num_workers + 1):
            proc = self.ctx.Process(
                target=worker_loop,
                args=(i, self.num_workers, self.config_path, self.task_queue, self.result_queue, self.stop_event, self.headless),
            )
            proc.start()
            self.workers.append(proc)

    def eval_genomes(self, genomes, config):
        self.generation_num += 1
        total = len(genomes)
        print(f"\n================ ПОКОЛЕНИЕ {self.generation_num} (ботов: {total}) ================", flush=True)
        genome_map = {}
        for genome_id, genome in genomes:
            genome_map[genome_id] = genome
            self.task_queue.put((genome_id, genome))
        completed = 0
        while completed < total:
            try:
                worker_id, genome_id, fitness, scores = self.result_queue.get(timeout=0.5)
            except queue.Empty:
                if not any(w.is_alive() for w in self.workers):
                    raise RuntimeError("Все воркеры завершились неожиданно.")
                continue
            completed += 1
            genome = genome_map[genome_id]
            genome.fitness = fitness
            scores_str = ", ".join(f"{s:.0f}" for s in scores)
            print(f"[Пок. {self.generation_num} | {completed}/{total} | Воркер #{worker_id} | ID={genome_id}] Забеги: [{scores_str}] -> Fitness: {fitness:.1f}", flush=True)
            if fitness > self.best_overall_fitness:
                self.best_overall_fitness = fitness
                self.best_overall_genome = copy.deepcopy(genome)
                save_genome(genome, self.model_path)

    def close(self):
        print("Закрываем все браузеры...", flush=True)
        self.stop_event.set()
        for q in (self.task_queue, self.result_queue):
            while True:
                try:
                    q.get_nowait()
                except Exception:
                    break
        for _ in self.workers:
            try:
                self.task_queue.put_nowait(None)
            except Exception:
                pass
        for proc in self.workers:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, CONFIG_FILE)
    model_path = os.path.join(base_dir, MODEL_FILE)
    checkpoint_dir = os.path.join(base_dir, CHECKPOINT_DIR)
    if not os.path.exists(config_path):
        print(f"Ошибка: не найден файл настроек '{CONFIG_FILE}' рядом со скриптом!")
        return
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    if config.genome_config.num_inputs != 7 or config.genome_config.num_outputs != 2:
        print(f"Ошибка в {CONFIG_FILE}: нужно num_inputs = 7 и num_outputs = 2")
        return
    if TRAIN_MODE:
        os.makedirs(checkpoint_dir, exist_ok=True)
        pop = None
        if RESUME_TRAINING:
            latest_ckpt = find_latest_checkpoint(checkpoint_dir)
            if latest_ckpt:
                restored = neat.Checkpointer.restore_checkpoint(latest_ckpt)
                if (restored.config.genome_config.num_inputs == config.genome_config.num_inputs
                        and restored.config.genome_config.num_outputs == config.genome_config.num_outputs):
                    print(f"Продолжаем обучение с последнего чекпоинта: {latest_ckpt}")
                    pop = restored
                    pop.config.fitness_threshold = config.fitness_threshold
                else:
                    print("Чекпоинт несовместим с новым конфигом — начинаем новую популяцию.")
            if pop is None and os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    best_genome = pickle.load(f)
                if is_genome_compatible(best_genome, config):
                    print(f"Создаём популяцию на основе лучшего бота: {model_path}")
                    pop = neat.Population(config)
                    seed_population_from_genome(pop, config, best_genome)
                else:
                    print(f"Файл {model_path} несовместим — начинаем с нуля.")
            if pop is None:
                print("Сохранённых данных v2 нет. Начинаем обучение с нуля.")
                pop = neat.Population(config)
        else:
            print("RESUME_TRAINING = False -> очищаем чекпоинты и начинаем с нуля.")
            clear_checkpoints(checkpoint_dir)
            pop = neat.Population(config)
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(RollingCheckpointer(generation_interval=1, filename_prefix=os.path.join(checkpoint_dir, CHECKPOINT_PREFIX), keep_last=KEEP_CHECKPOINTS))
        actual_workers = max(1, min(NUM_WORKERS, pop.config.pop_size))
        evaluator = ParallelDinoEvaluator(num_workers=actual_workers, config_path=config_path, model_path=model_path, headless=HEADLESS, start_generation=pop.generation)
        if RESUME_TRAINING and os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    prev_best = pickle.load(f)
                if is_genome_compatible(prev_best, config):
                    prev_fitness = getattr(prev_best, "fitness", None)
                    if prev_fitness is not None:
                        evaluator.best_overall_fitness = float(prev_fitness)
                        evaluator.best_overall_genome = prev_best
                        print(f"Текущий сохранённый рекорд: {prev_fitness:.1f} ({MODEL_FILE})")
            except Exception as e:
                print(f"Не удалось прочитать рекорд из {MODEL_FILE}: {e}")
        try:
            evaluator.start()
            winner = pop.run(evaluator.eval_genomes, GENERATIONS)
            print("\nОБУЧЕНИЕ ЗАВЕРШЕНО!")
            if winner and (winner.fitness or 0) > evaluator.best_overall_fitness:
                save_genome(winner, model_path)
        except KeyboardInterrupt:
            print("\nОбучение остановлено вручную (Ctrl+C).")
        finally:
            evaluator.close()
    else:
        if not os.path.exists(model_path):
            print(f"Файл '{model_path}' не найден! Сначала обучите бота с TRAIN_MODE = True.")
            return
        with open(model_path, "rb") as f:
            winner = pickle.load(f)
        if not is_genome_compatible(winner, config):
            print(f"Ошибка: бот в '{model_path}' несовместим с {CONFIG_FILE} (7 входов / 2 выхода).")
            return
        print(f"Загружен обученный бот из {model_path}")
        net_data = serialize_network(winner, config)
        with sync_playwright() as p:
            browser, page = create_browser_page(p, headless=False, worker_id=1, total_workers=1)
            try:
                for game_idx in range(1, TEST_GAMES + 1):
                    score = play_one_game(page, net_data, speedup=1, max_score=0.0, start_speed=6.0)
                    print(f"Тестовая игра №{game_idx} — Счёт: {score:.1f}")
                    time.sleep(0.5)
                input("\nНажми Enter, чтобы закрыть браузер...")
            except KeyboardInterrupt:
                print("\nПросмотр остановлен вручную.")
            finally:
                browser.close()


if __name__ == "__main__":
    main()
