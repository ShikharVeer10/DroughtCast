param(
  [Parameter(Mandatory=$true)][string]$Url,
  [Parameter(Mandatory=$true)][string]$OutFile
)
$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromMinutes(5)
$head = $client.SendAsync([System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Head, $Url)).GetAwaiter().GetResult()
$length = [long]$head.Content.Headers.ContentLength
if (-not $length) { throw "No content length for $Url" }
$chunk = 16MB
$stream = [System.IO.File]::Open($OutFile, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
try {
  for ($start = 0L; $start -lt $length; $start += $chunk) {
    $end = [Math]::Min($start + $chunk - 1, $length - 1)
    $request = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Get, $Url)
    $request.Headers.Range = [System.Net.Http.Headers.RangeHeaderValue]::new($start, $end)
    $response = $client.SendAsync($request).GetAwaiter().GetResult()
    $response.EnsureSuccessStatusCode()
    $bytes = $response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
    if ($bytes.LongLength -ne ($end - $start + 1)) { throw "Short range ${start}-${end}: received $($bytes.LongLength) bytes" }
    $stream.Write($bytes, 0, $bytes.Length)
  }
} finally {
  $stream.Dispose()
  $client.Dispose()
}
if ((Get-Item -LiteralPath $OutFile).Length -ne $length) { throw "Incomplete file $OutFile" }
