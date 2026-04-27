<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=utf-8');

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode([
        'ok' => false,
        'error' => 'Method not allowed. Use POST.'
    ]);
    exit;
}

/**
 * Configure this token in your web server environment:
 *   SetEnv METRICS_PUSH_TOKEN "your-strong-secret"
 */
$expectedToken = getenv('METRICS_PUSH_TOKEN') ?: '';
if ($expectedToken === '') {
    http_response_code(500);
    echo json_encode([
        'ok' => false,
        'error' => 'Server is missing METRICS_PUSH_TOKEN configuration.'
    ]);
    exit;
}

function getBearerToken(): ?string
{
    $headers = [];

    if (function_exists('getallheaders')) {
        $headers = getallheaders();
    }

    if (!empty($headers['Authorization'])) {
        $auth = trim((string)$headers['Authorization']);
        if (stripos($auth, 'Bearer ') === 0) {
            return trim(substr($auth, 7));
        }
        return $auth;
    }

    if (!empty($_SERVER['HTTP_AUTHORIZATION'])) {
        $auth = trim((string)$_SERVER['HTTP_AUTHORIZATION']);
        if (stripos($auth, 'Bearer ') === 0) {
            return trim(substr($auth, 7));
        }
        return $auth;
    }

    return null;
}

$providedToken = getBearerToken();
if ($providedToken === null || $providedToken === '') {
    $providedToken = isset($_POST['token']) ? trim((string)$_POST['token']) : '';
}

if ($providedToken === '' || !hash_equals($expectedToken, $providedToken)) {
    http_response_code(401);
    echo json_encode([
        'ok' => false,
        'error' => 'Unauthorized: invalid or missing token.'
    ]);
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
    echo json_encode([
        'ok' => false,
        'error' => 'No JSON payload received.'
    ]);
    exit;
}

$decoded = json_decode($jsonPayload, true);
if (!is_array($decoded)) {
    http_response_code(400);
    echo json_encode([
        'ok' => false,
        'error' => 'Invalid JSON payload.'
    ]);
    exit;
}

$targetPath = __DIR__ . DIRECTORY_SEPARATOR . 'latest_metrics.json';
$encoded = json_encode($decoded, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);

if ($encoded === false) {
    http_response_code(500);
    echo json_encode([
        'ok' => false,
        'error' => 'Unable to encode JSON for storage.'
    ]);
    exit;
}

$bytes = @file_put_contents($targetPath, $encoded . PHP_EOL, LOCK_EX);
if ($bytes === false) {
    http_response_code(500);
    echo json_encode([
        'ok' => false,
        'error' => 'Failed to write latest_metrics.json.'
    ]);
    exit;
}

http_response_code(200);
echo json_encode([
    'ok' => true,
    'message' => 'Metrics saved successfully.',
    'saved_bytes' => $bytes,
    'saved_to' => basename($targetPath),
    'timestamp' => gmdate('c')
]);

