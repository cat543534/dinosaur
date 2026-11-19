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
NUM_WORKERS = 15         # Сколько копий игры запускать одновременно
HEADLESS = True        # False = видеть окна сеткой, True = скрытый режим (быстрее)

# ---------- НАСТРОЙКИ NEAT И СОХРАНЕНИЯ ----------
CONFIG_FILE = "config-neat.txt"
MODEL_FILE = "best_dino_genome.pkl"
CHECKPOINT_DIR = "checkpoints"          # Папка для чекпоинтов всей популяции
CHECKPOINT_PREFIX = "neat-checkpoint-"
KEEP_CHECKPOINTS = 5                    # Сколько последних чекпоинтов хранить

TRAIN_MODE = True       # True = обучение NEAT, False = запуск лучшего сохранённого бота
RESUME_TRAINING = True  # True = продолжать с чекпоинта / лучшего бота, False = начать с нуля
GENERATIONS = 30        # Сколько поколений обучать за этот запуск
TEST_GAMES = 5          # Сколько игр сыграть в режиме TRAIN_MODE = False
STEP_DELAY = 0.02       # Частота опроса (50 раз/сек)
MAX_STEPS = 30000

# ---------- НАСТРОЙКИ ПРИСЕДА И ПРЫЖКА ----------
DUCK_KEY = "ArrowDown"
JUMP_START_TIMEOUT = 0.5
JUMP_LAND_TIMEOUT = 3.0
POLL_INTERVAL = 0.01

# JS-скрипт, который запрещает игре ставиться на паузу при потере фокуса окна
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


def save_genome(genome, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(genome, f)
    print(f"--> Новый рекорд! Геном сохранён в {filepath}")


# ---------- РАБОТА С ЧЕКПОИНТАМИ И ЗАСЕЛЕНИЕМ ПОПУЛЯЦИИ ----------

def find_latest_checkpoint(checkpoint_dir, prefix=CHECKPOINT_PREFIX):
    """Ищет чекпоинт с максимальным номером поколения."""
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
    """Удаляет старые чекпоинты (используется при RESUME_TRAINING = False)."""
    if not os.path.isdir(checkpoint_dir):
        return
    for name in os.listdir(checkpoint_dir):
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            try:
                os.remove(os.path.join(checkpoint_dir, name))
            except OSError:
                pass


class RollingCheckpointer(neat.Checkpointer):
    """Сохраняет всю популяцию после каждого поколения и хранит только последние N чекпоинтов."""

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


def seed_population_from_genome(pop, config, best_genome):
    """Заполняет популяцию копиями лучшего генома и его мутациями."""
    genome_cfg = config.genome_config

    # Сдвигаем счётчик ID нейронов, чтобы не было конфликтов при мутации добавления узла
    max_node_id = max(best_genome.nodes.keys(), default=0)
    min_next_node = max(max_node_id + 1, genome_cfg.num_outputs)
    genome_cfg.node_indexer = itertools.count(min_next_node)

    # Сдвигаем счётчик инноваций, если он есть в текущей версии neat-python
    tracker = getattr(genome_cfg, "innovation_tracker", None)
    if tracker is not None and best_genome.connections:
        max_innov = max(
            (getattr(c, "innovation", 0) or 0 for c in best_genome.connections.values()),
            default=0,
        )
        next_innov = getattr(tracker, "next_innovation", None)
        if isinstance(next_innov, int) and next_innov <= max_innov:
            tracker.next_innovation = max_innov + 1

    # Первые 2 бота — точные копии лучшего, остальные — его мутации
    num_exact = min(2, len(pop.population))
    new_population = {}

    for idx, gid in enumerate(list(pop.population.keys())):
        g = copy.deepcopy(best_genome)
        g.key = gid
        g.fitness = None

        if idx >= num_exact:
            g.mutate(genome_cfg)
            # Каждому третьему даём двойную мутацию для большего разнообразия поиска
            if idx % 3 == 0:
                g.mutate(genome_cfg)

        new_population[gid] = g

    pop.population = new_population
    pop.species = config.species_set_type(config.species_set_config, pop.reporters)
    pop.species.speciate(config, pop.population, pop.generation)


# ---------- УПРАВЛЕНИЕ ДИНОЗАВРОМ ----------

def get_trex_info(page):
    """Читаем состояние динозавра прямо из игры."""
    try:
        return page.evaluate(
            """
            () => {
                const tRex = window.Runner?.instance_?.tRex;
                if (!tRex) return null;
                return {
                    status: tRex.status,
                    jumping: !!tRex.jumping,
                };
            }
            """
        )
    except Exception:
        return None


def duck(page):
    try:
        page.keyboard.down(DUCK_KEY)
    except Exception:
        pass


def stand_up(page):
    try:
        page.keyboard.up(DUCK_KEY)
    except Exception:
        pass


def ensure_duck(page, timeout=1.5, stop_event=None):
    """Сажаем динозавра в присед и дожидаемся статуса DUCKING."""
    deadline = time.time() + timeout
    duck(page)

    while time.time() < deadline:
        if stop_event and stop_event.is_set():
            return False

        time.sleep(0.02)
        info = get_trex_info(page)

        if info and info["status"] == "DUCKING":
            return True
        if info and info["status"] == "CRASHED":
            return False

        duck(page)

    return False


def do_jump(page, stop_event=None):
    """Прыжок из приседа: встать -> прыгнуть -> дождаться приземления -> снова присесть."""
    info = get_trex_info(page)
    if not info or info["status"] == "CRASHED":
        return

    stand_up(page)
    page.keyboard.press("Space")

    deadline = time.time() + JUMP_START_TIMEOUT
    while time.time() < deadline:
        if stop_event and stop_event.is_set():
            return
        info = get_trex_info(page)
        if info and (info["jumping"] or info["status"] == "CRASHED"):
            break
        time.sleep(POLL_INTERVAL)

    deadline = time.time() + JUMP_LAND_TIMEOUT
    while time.time() < deadline:
        if stop_event and stop_event.is_set():
            return
        info = get_trex_info(page)
        if info and (
            info["status"] == "CRASHED"
            or (not info["jumping"] and info["status"] != "JUMPING")
        ):
            break
        time.sleep(POLL_INTERVAL)

    info = get_trex_info(page)
    if info and info["status"] != "CRASHED":
        ensure_duck(page, timeout=0.5, stop_event=stop_event)


# ---------- ПОЛУЧЕНИЕ ДАННЫХ ДЛЯ НЕЙРОСЕТИ ----------

def get_game_data(page):
    """Возвращает кортеж: (inputs, score, status)."""
    try:
        result = page.evaluate(
            """
            () => {
                const runner = window.Runner?.instance_;
                if (!runner) return { status: "NO_RUNNER" };

                const score = (runner.distanceRan || 0) * 0.025;

                if (runner.crashed) {
                    return { status: "DEAD", score: score };
                }

                if (!runner.playing) {
                    return { status: "NOT_PLAYING", score: score };
                }

                const dino = runner.tRex;
                const obstacles = runner.horizon?.obstacles || [];
                const speed = runner.currentSpeed || 6.0;

                let nextObs = null;
                for (const obs of obstacles) {
                    const obsWidth = obs.width || (obs.typeConfig ? obs.typeConfig.width * (obs.size || 1) : 25);
                    if (obs.xPos + obsWidth > dino.xPos) {
                        nextObs = {
                            distance: obs.xPos - dino.xPos,
                            width: obsWidth,
                            yPos: obs.yPos || 100
                        };
                        break;
                    }
                }

                if (!nextObs) {
                    return {
                        status: "OK",
                        distance: 600.0,
                        speed: speed,
                        obs_width: 0.0,
                        obs_y: 0.0,
                        score: score
                    };
                }

                return {
                    status: "OK",
                    distance: nextObs.distance,
                    speed: speed,
                    obs_width: nextObs.width,
                    obs_y: nextObs.yPos,
                    score: score
                };
            }
            """
        )

        status = result["status"]
        score = float(result.get("score", 0.0))

        if status != "OK":
            return None, score, status

        inputs = (
            result["distance"] / 600.0,
            result["speed"] / 15.0,
            result["obs_width"] / 100.0,
            result["obs_y"] / 150.0,
        )
        return inputs, score, "OK"

    except Exception as error:
        return None, 0.0, f"ERROR: {error}"


def start_game(page):
    """Запускает новую игру (быстрый рестарт через JS, если это не первый запуск)."""
    stand_up(page)

    restarted = page.evaluate(
        """
        () => {
            const runner = window.Runner?.instance_;
            if (runner && runner.crashed) {
                runner.restart();
                return true;
            }
            return false;
        }
        """
    )

    if restarted:
        time.sleep(0.15)
    else:
        canvas = page.locator("canvas")
        canvas.click()
        time.sleep(0.3)
        page.keyboard.press("Space")
        time.sleep(1.0)

    inputs, score, status = get_game_data(page)

    if inputs is None and restarted:
        time.sleep(0.8)
        page.keyboard.press("Space")
        time.sleep(0.5)
        inputs, score, status = get_game_data(page)

    return inputs, score, status


# ---------- ИГРОВОЙ ЦИКЛ ДЛЯ ОДНОЙ НЕЙРОСЕТИ ----------

def play_one_game(page, net, stop_event=None):
    """Прогоняет одну игру для нейросети и возвращает набранный счёт (fitness)."""
    inputs, score, status = start_game(page)

    if inputs is None:
        page.reload(wait_until="networkidle")
        time.sleep(2.0)
        inputs, score, status = start_game(page)
        if inputs is None:
            return 0.0

    ensure_duck(page, stop_event=stop_event)

    last_score = 0.0

    for _ in range(MAX_STEPS):
        if stop_event and stop_event.is_set():
            break

        output = net.activate(inputs)

        if output[0] > 0.5:
            do_jump(page, stop_event=stop_event)

        time.sleep(STEP_DELAY)

        next_inputs, score, status = get_game_data(page)
        if score > last_score:
            last_score = score

        if next_inputs is None:
            break

        inputs = next_inputs

    stand_up(page)
    return last_score


# ---------- ОТДЕЛЬНЫЙ ПРОЦЕСС-ВОРКЕР ----------

def create_browser_page(p, headless=False, worker_id=1, total_workers=1):
    """Запускает Chromium с нужными флагами и размещает окно в сетке."""
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

    browser = p.chromium.launch(
        channel="chromium",
        headless=headless,
        args=launch_args,
    )

    viewport = {"width": 620, "height": 260} if total_workers > 1 else {"width": 800, "height": 600}
    page = browser.new_page(viewport=viewport)

    page.add_init_script(ANTI_PAUSE_JS)
    page.goto(GAME_URL, wait_until="networkidle", timeout=60000)
    time.sleep(2.0)

    return browser, page


def worker_loop(worker_id, total_workers, config_path, task_queue, result_queue, stop_event, headless):
    """Держит браузер открытым и тестирует присланных ботов."""
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
                p,
                headless=headless,
                worker_id=worker_id,
                total_workers=total_workers,
            )
            print(f"[Воркер #{worker_id}] Браузер готов к работе.")

            try:
                while not stop_event.is_set():
                    try:
                        task = task_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if task is None:
                        break

                    genome_id, genome = task
                    net = neat.nn.FeedForwardNetwork.create(genome, config)

                    try:
                        fitness = play_one_game(page, net, stop_event=stop_event)
                    except Exception as e:
                        print(f"[Воркер #{worker_id}] Ошибка в игре (ID={genome_id}): {e}")
                        fitness = 0.0

                    if not stop_event.is_set():
                        result_queue.put((worker_id, genome_id, fitness))
            finally:
                try:
                    browser.close()
                except Exception:
                    pass
    except Exception as e:
        print(f"[Воркер #{worker_id}] Критическая ошибка запуска: {e}")
    finally:
        task_queue.cancel_join_thread()
        result_queue.cancel_join_thread()


# ---------- МЕНЕДЖЕР ПАРАЛЛЕЛЬНОГО ОБУЧЕНИЯ ----------

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
        print(f"Запускаем {self.num_workers} параллельных браузеров (headless={self.headless})...")
        for i in range(1, self.num_workers + 1):
            proc = self.ctx.Process(
                target=worker_loop,
                args=(
                    i,
                    self.num_workers,
                    self.config_path,
                    self.task_queue,
                    self.result_queue,
                    self.stop_event,
                    self.headless,
                ),
            )
            proc.start()
            self.workers.append(proc)

    def eval_genomes(self, genomes, config):
        self.generation_num += 1
        total = len(genomes)
        print(f"\n================ ПОКОЛЕНИЕ {self.generation_num} (ботов: {total}) ================")

        genome_map = {}
        for genome_id, genome in genomes:
            genome_map[genome_id] = genome
            self.task_queue.put((genome_id, genome))

        completed = 0
        while completed < total:
            try:
                worker_id, genome_id, fitness = self.result_queue.get(timeout=0.5)
            except queue.Empty:
                if not any(w.is_alive() for w in self.workers):
                    raise RuntimeError("Все воркеры завершились неожиданно.")
                continue

            completed += 1
            genome = genome_map[genome_id]
            genome.fitness = fitness

            print(
                f"[Поколение {self.generation_num} | Готово {completed}/{total} | "
                f"Воркер #{worker_id} | ID={genome_id}] Счёт: {fitness:.1f}"
            )

            if fitness > self.best_overall_fitness:
                self.best_overall_fitness = fitness
                self.best_overall_genome = copy.deepcopy(genome)
                save_genome(genome, self.model_path)

    def close(self):
        print("Закрываем все браузеры...")
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

    if TRAIN_MODE:
        os.makedirs(checkpoint_dir, exist_ok=True)
        pop = None

        if RESUME_TRAINING:
            latest_ckpt = find_latest_checkpoint(checkpoint_dir)
            if latest_ckpt:
                print(f"Продолжаем обучение с последнего чекпоинта: {latest_ckpt}")
                pop = neat.Checkpointer.restore_checkpoint(latest_ckpt)
                pop.config.fitness_threshold = config.fitness_threshold
            elif os.path.exists(model_path):
                print(f"Чекпоинты не найдены. Создаём популяцию на основе лучшего бота: {model_path}")
                with open(model_path, "rb") as f:
                    best_genome = pickle.load(f)
                pop = neat.Population(config)
                seed_population_from_genome(pop, config, best_genome)
            else:
                print("Сохранённых данных нет. Начинаем обучение с нуля.")
                pop = neat.Population(config)
        else:
            print("RESUME_TRAINING = False -> очищаем чекпоинты и начинаем с нуля.")
            clear_checkpoints(checkpoint_dir)
            pop = neat.Population(config)

        # Добавляем репортеры и автосохранение чекпоинта после каждого поколения
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(
            RollingCheckpointer(
                generation_interval=1,
                filename_prefix=os.path.join(checkpoint_dir, CHECKPOINT_PREFIX),
                keep_last=KEEP_CHECKPOINTS,
            )
        )

        actual_workers = max(1, min(NUM_WORKERS, pop.config.pop_size))

        evaluator = ParallelDinoEvaluator(
            num_workers=actual_workers,
            config_path=config_path,
            model_path=model_path,
            headless=HEADLESS,
            start_generation=pop.generation,
        )

        # Загружаем прошлый рекорд, чтобы новый запуск не перезаписал его слабым ботом
        if RESUME_TRAINING and os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    prev_best = pickle.load(f)
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
            if evaluator.best_overall_genome is not None:
                print(
                    f"Лучший результат ({evaluator.best_overall_fitness:.1f}) "
                    f"сохранён в {MODEL_FILE}."
                )
        finally:
            evaluator.close()

    else:
        if not os.path.exists(model_path):
            print(f"Файл '{model_path}' не найден! Сначала обучите бота с TRAIN_MODE = True.")
            return

        with open(model_path, "rb") as f:
            winner = pickle.load(f)

        print(f"Загружен обученный бот из {model_path}")
        net = neat.nn.FeedForwardNetwork.create(winner, config)

        with sync_playwright() as p:
            browser, page = create_browser_page(p, headless=False, worker_id=1, total_workers=1)
            try:
                for game_idx in range(1, TEST_GAMES + 1):
                    score = play_one_game(page, net)
                    print(f"Тестовая игра №{game_idx} — Счёт: {score:.1f}")
                    time.sleep(0.5)
                input("\nНажми Enter, чтобы закрыть браузер...")
            finally:
                browser.close()


if __name__ == "__main__":
    main()