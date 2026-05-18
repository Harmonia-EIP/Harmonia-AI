<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=utf-8');
header('Access-Control-Allow-Origin: *');
header('Cache-Control: no-store');

function readJsonArray(string $path): array
{
    if (!is_file($path)) {
        return [];
    }
    $raw = @file_get_contents($path);
    if (!is_string($raw) || trim($raw) === '') {
        return [];
    }
    $decoded = json_decode($raw, true);
    return is_array($decoded) ? $decoded : [];
}

function readJsonObject(string $path): array
{
    return readJsonArray($path);
}

$baseDir = __DIR__;
$action = isset($_GET['action']) ? preg_replace('/[^a-z0-9_-]/i', '', (string)$_GET['action']) : 'summary';
$action = $action ?: 'summary';

function respond($payload, int $status = 200): void
{
    http_response_code($status);
    echo json_encode($payload, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
    exit;
}

if ($action === 'probe') {
    // Safe diagnostic: tells whether receiver can find a token, never echoes its value.
    $sources = [
        'getenv' => is_string(getenv('METRICS_PUSH_TOKEN')) && getenv('METRICS_PUSH_TOKEN') !== '',
        '_SERVER' => isset($_SERVER['METRICS_PUSH_TOKEN']) && $_SERVER['METRICS_PUSH_TOKEN'] !== '',
        '_SERVER_REDIRECT' => isset($_SERVER['REDIRECT_METRICS_PUSH_TOKEN']) && $_SERVER['REDIRECT_METRICS_PUSH_TOKEN'] !== '',
        '_ENV' => isset($_ENV['METRICS_PUSH_TOKEN']) && $_ENV['METRICS_PUSH_TOKEN'] !== '',
        'dotenv_file' => is_readable(__DIR__ . '/.env'),
    ];
    $writable = is_writable(__DIR__) && (is_writable(__DIR__ . '/events') || @mkdir(__DIR__ . '/events', 0775, true));
    respond([
        'ok' => true,
        'token_sources' => $sources,
        'token_resolved' => in_array(true, $sources, true),
        'events_dir_writable' => $writable,
        'php_version' => PHP_VERSION,
    ]);
}

if ($action === 'summary') {
    $index = readJsonArray($baseDir . '/index.json');
    $models = readJsonArray($baseDir . '/models.json');
    $training = array_values(array_filter($index, fn($e) => ($e['kind'] ?? '') === 'training'));
    $generations = array_values(array_filter($index, fn($e) => ($e['kind'] ?? '') === 'generation'));
    $latestTraining = end($training) ?: null;
    $latestGeneration = end($generations) ?: null;
    $latest = readJsonObject($baseDir . '/latest_metrics.json');
    respond([
        'ok' => true,
        'counts' => [
            'events' => count($index),
            'training' => count($training),
            'generations' => count($generations),
            'models' => count($models),
        ],
        'latest_training_summary' => $latestTraining,
        'latest_generation_summary' => $latestGeneration,
        'latest_training_payload' => $latest,
    ]);
}

if ($action === 'index') {
    $index = readJsonArray($baseDir . '/index.json');
    $limit = isset($_GET['limit']) ? max(1, min(2000, (int)$_GET['limit'])) : 200;
    $kind = isset($_GET['kind']) ? preg_replace('/[^a-z0-9_-]/i', '', (string)$_GET['kind']) : '';
    $model = isset($_GET['model']) ? (string)$_GET['model'] : '';
    if ($kind !== '') {
        $index = array_values(array_filter($index, fn($e) => ($e['kind'] ?? '') === $kind));
    }
    if ($model !== '') {
        $index = array_values(array_filter($index, fn($e) => ($e['model_version'] ?? '') === $model));
    }
    $index = array_reverse($index);
    if ($limit < count($index)) {
        $index = array_slice($index, 0, $limit);
    }
    respond(['ok' => true, 'events' => $index]);
}

if ($action === 'models') {
    $models = readJsonArray($baseDir . '/models.json');
    usort($models, function ($a, $b) {
        return strcmp((string)($b['last_seen_at'] ?? ''), (string)($a['last_seen_at'] ?? ''));
    });
    respond(['ok' => true, 'models' => $models]);
}

if ($action === 'history') {
    $history = readJsonArray($baseDir . '/history_metrics.json');
    $model = isset($_GET['model']) ? (string)$_GET['model'] : '';
    if ($model !== '') {
        $history = array_values(array_filter($history, fn($e) => ($e['model_version'] ?? '') === $model));
    }
    respond(['ok' => true, 'history' => $history]);
}

if ($action === 'presets') {
    $presets = readJsonArray($baseDir . '/presets.json');
    $model = isset($_GET['model']) ? (string)$_GET['model'] : '';
    $search = isset($_GET['q']) ? trim((string)$_GET['q']) : '';
    $limit = isset($_GET['limit']) ? max(1, min(1000, (int)$_GET['limit'])) : 200;
    if ($model !== '') {
        $presets = array_values(array_filter($presets, fn($e) => ($e['model_version'] ?? '') === $model));
    }
    if ($search !== '') {
        $needle = strtolower($search);
        $presets = array_values(array_filter($presets, function ($e) use ($needle) {
            $prompt = strtolower((string)($e['prompt'] ?? ''));
            return strpos($prompt, $needle) !== false;
        }));
    }
    $presets = array_reverse($presets);
    if ($limit < count($presets)) {
        $presets = array_slice($presets, 0, $limit);
    }
    respond(['ok' => true, 'presets' => $presets]);
}

if ($action === 'performance') {
    $perf = readJsonArray($baseDir . '/performance.json');
    $limit = isset($_GET['limit']) ? max(1, min(1000, (int)$_GET['limit'])) : 200;
    $kind = isset($_GET['kind']) ? preg_replace('/[^a-z0-9_-]/i', '', (string)$_GET['kind']) : '';
    if ($kind !== '') {
        $perf = array_values(array_filter($perf, fn($e) => ($e['kind'] ?? '') === $kind));
    }
    $perf = array_reverse($perf);
    if ($limit < count($perf)) {
        $perf = array_slice($perf, 0, $limit);
    }
    // Lightweight aggregates over the window so the dashboard can render
    // averages and peaks without re-walking everything client-side.
    $cpuValues = [];
    $ramValues = [];
    $gpuValues = [];
    $latencies = [];
    foreach ($perf as $entry) {
        $sys = $entry['system_metrics'] ?? null;
        if (is_array($sys)) {
            if (isset($sys['cpu_percent']) && is_numeric($sys['cpu_percent'])) $cpuValues[] = (float)$sys['cpu_percent'];
            if (isset($sys['ram_percent']) && is_numeric($sys['ram_percent'])) $ramValues[] = (float)$sys['ram_percent'];
            $gpu = $sys['gpu_mps'] ?? null;
            if (is_array($gpu) && isset($gpu['allocated_mb']) && is_numeric($gpu['allocated_mb'])) {
                $gpuValues[] = (float)$gpu['allocated_mb'];
            }
        }
        if (isset($entry['inference_latency_ms']) && is_numeric($entry['inference_latency_ms'])) {
            $latencies[] = (float)$entry['inference_latency_ms'];
        }
    }
    $aggregate = function(array $values): array {
        if (!$values) return ['count' => 0];
        $sum = array_sum($values);
        return [
            'count' => count($values),
            'avg' => round($sum / count($values), 3),
            'max' => round(max($values), 3),
            'min' => round(min($values), 3),
            'last' => round($values[0], 3),
        ];
    };
    respond([
        'ok' => true,
        'entries' => $perf,
        'aggregates' => [
            'cpu_percent' => $aggregate($cpuValues),
            'ram_percent' => $aggregate($ramValues),
            'gpu_mps_allocated_mb' => $aggregate($gpuValues),
            'inference_latency_ms' => $aggregate($latencies),
        ],
    ]);
}

if ($action === 'event') {
    $file = isset($_GET['file']) ? (string)$_GET['file'] : '';
    if ($file === '' || strpos($file, '..') !== false) {
        respond(['ok' => false, 'error' => 'Invalid file parameter.'], 400);
    }
    $file = ltrim($file, '/');
    if (strpos($file, 'events/') !== 0) {
        respond(['ok' => false, 'error' => 'Only event files are exposed.'], 400);
    }
    $resolved = realpath($baseDir . '/' . $file);
    $eventsRoot = realpath($baseDir . '/events');
    if ($resolved === false || $eventsRoot === false || strpos($resolved, $eventsRoot) !== 0) {
        respond(['ok' => false, 'error' => 'Event not found.'], 404);
    }
    $payload = json_decode((string)file_get_contents($resolved), true);
    respond(['ok' => true, 'event' => $payload]);
}

respond(['ok' => false, 'error' => "Unknown action: {$action}"], 400);
