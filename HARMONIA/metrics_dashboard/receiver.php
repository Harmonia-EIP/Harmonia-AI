<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=utf-8');

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['ok' => false, 'error' => 'Method not allowed. Use POST.']);
    exit;
}

$expectedToken = getenv('METRICS_PUSH_TOKEN') ?: ($_SERVER['METRICS_PUSH_TOKEN'] ?? null) ?: ($_SERVER['REDIRECT_METRICS_PUSH_TOKEN'] ?? null) ?: '';
if ($expectedToken === '') {
    http_response_code(500);
    echo json_encode(['ok' => false, 'error' => 'Server is missing METRICS_PUSH_TOKEN configuration.']);
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

$metrics = extractMetrics($decoded);
$latestPayload = $decoded;
$latestPath = __DIR__ . DIRECTORY_SEPARATOR . 'latest_metrics.json';
$encodedLatest = json_encode($latestPayload, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
if ($encodedLatest === false || file_put_contents($latestPath, $encodedLatest . PHP_EOL, LOCK_EX) === false) {
    http_response_code(500);
    echo json_encode(['ok' => false, 'error' => 'Failed to write latest_metrics.json.']);
    exit;
}

$historyPath = __DIR__ . DIRECTORY_SEPARATOR . 'history_metrics.json';
$history = [];
if (is_file($historyPath)) {
    $existing = @file_get_contents($historyPath);
    if (is_string($existing) && trim($existing) !== '') {
        $parsed = json_decode($existing, true);
        if (is_array($parsed)) {
            $history = $parsed;
        }
    }
}

$entry = [
    'timestamp' => $decoded['timestamp'] ?? gmdate('c'),
    'model_version' => $decoded['model_version'] ?? ($decoded['latest_evaluation_report']['model_version'] ?? 'unknown'),
    'model_hash' => $decoded['model_hash'] ?? ($decoded['latest_evaluation_report']['model_hash'] ?? 'unknown'),
    'loss' => $metrics['loss'] ?? $metrics['final_loss'] ?? null,
    'final_loss' => $metrics['final_loss'] ?? $metrics['loss'] ?? null,
    'mse' => $metrics['mse'] ?? null,
    'mae' => $metrics['mae'] ?? null,
    'continuous_mse' => $metrics['continuous_mse'] ?? null,
    'binary_bce' => $metrics['binary_bce'] ?? null,
    'metrics' => $metrics,
];
$history[] = $entry;
if (count($history) > 500) {
    $history = array_slice($history, -500);
}
$encodedHistory = json_encode($history, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
if ($encodedHistory !== false) {
    @file_put_contents($historyPath, $encodedHistory . PHP_EOL, LOCK_EX);
}

http_response_code(200);
echo json_encode([
    'ok' => true,
    'message' => 'Metrics saved successfully.',
    'saved_to' => basename($latestPath),
    'history_file' => basename($historyPath),
    'timestamp' => gmdate('c')
]);

