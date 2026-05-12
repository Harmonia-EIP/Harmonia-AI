<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=utf-8');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Authorization, Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(204);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['ok' => false, 'error' => 'Method not allowed. Use POST.']);
    exit;
}

function loadExpectedToken(): string
{
    $sources = [
        getenv('METRICS_PUSH_TOKEN'),
        $_SERVER['METRICS_PUSH_TOKEN'] ?? null,
        $_SERVER['REDIRECT_METRICS_PUSH_TOKEN'] ?? null,
        $_ENV['METRICS_PUSH_TOKEN'] ?? null,
    ];
    foreach ($sources as $candidate) {
        if (is_string($candidate) && $candidate !== '') {
            return $candidate;
        }
    }
    // .env fallback: same directory as receiver.php, protected by .htaccess.
    $envFile = __DIR__ . DIRECTORY_SEPARATOR . '.env';
    if (is_readable($envFile)) {
        $lines = @file($envFile, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
        if (is_array($lines)) {
            foreach ($lines as $line) {
                $line = trim($line);
                if ($line === '' || strpos($line, '#') === 0) {
                    continue;
                }
                if (strpos($line, 'METRICS_PUSH_TOKEN=') === 0) {
                    return trim(substr($line, strlen('METRICS_PUSH_TOKEN=')), " \t\"'");
                }
            }
        }
    }
    return '';
}

$expectedToken = loadExpectedToken();
if ($expectedToken === '') {
    http_response_code(500);
    echo json_encode([
        'ok' => false,
        'error' => 'Server is missing METRICS_PUSH_TOKEN configuration.',
        'hint' => 'Set Apache SetEnv METRICS_PUSH_TOKEN OR create a .env file (METRICS_PUSH_TOKEN=...) next to receiver.php.',
    ]);
    exit;
}

function getBearerToken(): ?string
{
    $headers = function_exists('getallheaders') ? getallheaders() : [];
    foreach (['Authorization', 'authorization'] as $headerName) {
        if (!empty($headers[$headerName])) {
            $auth = trim((string)$headers[$headerName]);
            return stripos($auth, 'Bearer ') === 0 ? trim(substr($auth, 7)) : $auth;
        }
    }
    if (!empty($_SERVER['HTTP_AUTHORIZATION'])) {
        $auth = trim((string)$_SERVER['HTTP_AUTHORIZATION']);
        return stripos($auth, 'Bearer ') === 0 ? trim(substr($auth, 7)) : $auth;
    }
    return null;
}

$providedToken = getBearerToken();
if ($providedToken === null || $providedToken === '') {
    $providedToken = isset($_POST['token']) ? trim((string)$_POST['token']) : '';
}
if ($providedToken === '' || !hash_equals($expectedToken, $providedToken)) {
    http_response_code(401);
    echo json_encode(['ok' => false, 'error' => 'Unauthorized: invalid or missing token.']);
    exit;
}

$jsonPayload = '';
if (isset($_FILES['metrics_file']) && is_uploaded_file($_FILES['metrics_file']['tmp_name'])) {
    $jsonPayload = (string)file_get_contents($_FILES['metrics_file']['tmp_name']);
} elseif (!empty($_POST['metrics_json'])) {
    $jsonPayload = (string)$_POST['metrics_json'];
} else {
    $jsonPayload = (string)file_get_contents('php://input');
}

if (trim($jsonPayload) === '') {
    http_response_code(400);
    echo json_encode(['ok' => false, 'error' => 'No JSON payload received.']);
    exit;
}

$decoded = json_decode($jsonPayload, true);
if (!is_array($decoded)) {
    http_response_code(400);
    echo json_encode(['ok' => false, 'error' => 'Invalid JSON payload.']);
    exit;
}

$kind = isset($_GET['kind']) ? (string)$_GET['kind'] : (string)($decoded['event_kind'] ?? 'training');
$kind = preg_replace('/[^a-z0-9_-]/i', '', strtolower($kind)) ?: 'training';

function extractMetrics(array $payload): array
{
    if (isset($payload['metrics']) && is_array($payload['metrics'])) {
        return $payload['metrics'];
    }
    if (isset($payload['latest_evaluation_report']['metrics']) && is_array($payload['latest_evaluation_report']['metrics'])) {
        return $payload['latest_evaluation_report']['metrics'];
    }
    if (isset($payload['latest_benchmark']['eval_metrics']) && is_array($payload['latest_benchmark']['eval_metrics'])) {
        return $payload['latest_benchmark']['eval_metrics'];
    }
    return [];
}

function slugify(string $value): string
{
    $value = preg_replace('/[^A-Za-z0-9._-]+/', '-', $value);
    $value = trim((string)$value, '-');
    if ($value === '') {
        $value = 'event';
    }
    return strtolower(substr($value, 0, 48));
}

function readJsonArray(string $path): array
{
    if (!is_file($path)) {
        return [];
    }
    $existing = @file_get_contents($path);
    if (!is_string($existing) || trim($existing) === '') {
        return [];
    }
    $parsed = json_decode($existing, true);
    return is_array($parsed) ? $parsed : [];
}

function writeJson(string $path, $payload): bool
{
    $encoded = json_encode($payload, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
    if ($encoded === false) {
        return false;
    }
    return @file_put_contents($path, $encoded . PHP_EOL, LOCK_EX) !== false;
}

$baseDir = __DIR__;
$eventsDir = $baseDir . DIRECTORY_SEPARATOR . 'events';
if (!is_dir($eventsDir)) {
    @mkdir($eventsDir, 0775, true);
}

$timestamp = $decoded['timestamp'] ?? gmdate('c');
$modelVersion = $decoded['model_version'] ?? ($decoded['latest_evaluation_report']['model_version'] ?? 'unknown');
$modelHash = $decoded['model_hash'] ?? ($decoded['latest_evaluation_report']['model_hash'] ?? 'unknown');
$metrics = extractMetrics($decoded);

$eventId = gmdate('Ymd_His') . '_' . substr(bin2hex(random_bytes(3)), 0, 6);
$eventFilename = sprintf('%s_%s_%s_%s.json', gmdate('Ymd_His'), slugify($kind), slugify((string)$modelVersion), substr(bin2hex(random_bytes(2)), 0, 4));
$eventPath = $eventsDir . DIRECTORY_SEPARATOR . $eventFilename;

$summary = [
    'id' => $eventId,
    'file' => 'events/' . $eventFilename,
    'kind' => $kind,
    'timestamp' => $timestamp,
    'received_at' => gmdate('c'),
    'model_version' => $modelVersion,
    'model_hash' => $modelHash,
];

if ($kind === 'training' && $metrics) {
    $summary['loss'] = $metrics['loss'] ?? $metrics['final_loss'] ?? null;
    $summary['final_loss'] = $metrics['final_loss'] ?? $metrics['loss'] ?? null;
    $summary['mse'] = $metrics['mse'] ?? null;
    $summary['mae'] = $metrics['mae'] ?? null;
    $summary['continuous_mse'] = $metrics['continuous_mse'] ?? null;
    $summary['binary_bce'] = $metrics['binary_bce'] ?? null;
    $summary['sample_count'] = $metrics['sample_count'] ?? null;
}

if ($kind === 'generation') {
    $summary['prompt'] = isset($decoded['prompt']) ? (string)$decoded['prompt'] : ($decoded['preset']['prompt'] ?? null);
    $summary['preset_name'] = $decoded['preset_name'] ?? null;
    if (isset($decoded['values']) && is_array($decoded['values'])) {
        $summary['value_count'] = count($decoded['values']);
    }
}

if ($kind === 'command') {
    $summary['command'] = $decoded['command'] ?? null;
    $summary['status'] = $decoded['status'] ?? null;
    $summary['duration_seconds'] = $decoded['duration_seconds'] ?? null;
}

if (!writeJson($eventPath, $decoded)) {
    http_response_code(500);
    echo json_encode(['ok' => false, 'error' => 'Failed to persist event payload.']);
    exit;
}

$indexPath = $baseDir . DIRECTORY_SEPARATOR . 'index.json';
$index = readJsonArray($indexPath);
$index[] = $summary;
if (count($index) > 5000) {
    $index = array_slice($index, -5000);
}
writeJson($indexPath, $index);

if ($kind === 'training') {
    $latestPath = $baseDir . DIRECTORY_SEPARATOR . 'latest_metrics.json';
    writeJson($latestPath, $decoded);

    $historyPath = $baseDir . DIRECTORY_SEPARATOR . 'history_metrics.json';
    $history = readJsonArray($historyPath);
    $history[] = [
        'timestamp' => $timestamp,
        'model_version' => $modelVersion,
        'model_hash' => $modelHash,
        'loss' => $summary['loss'] ?? null,
        'final_loss' => $summary['final_loss'] ?? null,
        'mse' => $summary['mse'] ?? null,
        'mae' => $summary['mae'] ?? null,
        'continuous_mse' => $summary['continuous_mse'] ?? null,
        'binary_bce' => $summary['binary_bce'] ?? null,
        'metrics' => $metrics,
        'event_file' => $summary['file'],
    ];
    if (count($history) > 500) {
        $history = array_slice($history, -500);
    }
    writeJson($historyPath, $history);
}

if ($kind === 'generation') {
    $presetsPath = $baseDir . DIRECTORY_SEPARATOR . 'presets.json';
    $presets = readJsonArray($presetsPath);
    $presets[] = $summary + [
        'prompt' => $summary['prompt'] ?? null,
        'values' => isset($decoded['values']) && is_array($decoded['values']) ? array_values($decoded['values']) : null,
        'parameters' => $decoded['parameters'] ?? null,
    ];
    if (count($presets) > 2000) {
        $presets = array_slice($presets, -2000);
    }
    writeJson($presetsPath, $presets);
}

$modelsPath = $baseDir . DIRECTORY_SEPARATOR . 'models.json';
$models = readJsonArray($modelsPath);
$found = false;
foreach ($models as &$entry) {
    if (($entry['model_version'] ?? null) === $modelVersion) {
        $entry['last_seen_at'] = gmdate('c');
        $entry['model_hash'] = $modelHash;
        $entry['event_count'] = ($entry['event_count'] ?? 0) + 1;
        if ($kind === 'training') {
            $entry['training_count'] = ($entry['training_count'] ?? 0) + 1;
            $entry['last_training_event'] = $summary['file'];
            $entry['last_loss'] = $summary['loss'] ?? $entry['last_loss'] ?? null;
            $entry['last_mse'] = $summary['mse'] ?? $entry['last_mse'] ?? null;
            $entry['last_mae'] = $summary['mae'] ?? $entry['last_mae'] ?? null;
        }
        if ($kind === 'generation') {
            $entry['generation_count'] = ($entry['generation_count'] ?? 0) + 1;
            $entry['last_generation_event'] = $summary['file'];
            $entry['last_prompt'] = $summary['prompt'] ?? $entry['last_prompt'] ?? null;
        }
        $found = true;
        break;
    }
}
unset($entry);
if (!$found) {
    $models[] = [
        'model_version' => $modelVersion,
        'model_hash' => $modelHash,
        'first_seen_at' => gmdate('c'),
        'last_seen_at' => gmdate('c'),
        'event_count' => 1,
        'training_count' => $kind === 'training' ? 1 : 0,
        'generation_count' => $kind === 'generation' ? 1 : 0,
        'last_training_event' => $kind === 'training' ? $summary['file'] : null,
        'last_generation_event' => $kind === 'generation' ? $summary['file'] : null,
        'last_loss' => $summary['loss'] ?? null,
        'last_mse' => $summary['mse'] ?? null,
        'last_mae' => $summary['mae'] ?? null,
        'last_prompt' => $summary['prompt'] ?? null,
    ];
}
writeJson($modelsPath, $models);

http_response_code(200);
echo json_encode([
    'ok' => true,
    'kind' => $kind,
    'event' => $summary,
    'timestamp' => gmdate('c'),
]);
