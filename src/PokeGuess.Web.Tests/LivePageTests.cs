using System.Net;

namespace PokeGuess.Web.Tests;

public sealed class LivePageTests : IDisposable
{
    private const string OrtScript = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.all.min.js";
    private readonly PokeGuessFactory _factory = new();
    private readonly HttpClient _client;

    public LivePageTests() => _client = _factory.CreateClient();
    public void Dispose() => _factory.Dispose();

    [Fact]
    public async Task Live_page_renders_start_button()
    {
        var res = await _client.GetAsync("/Live");
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        var html = await res.Content.ReadAsStringAsync();
        Assert.Contains("id=\"startBtn\"", html);
        Assert.Contains(OrtScript, html);
        Assert.Contains("id=\"resultPanel\"", html);
    }

    [Fact]
    public async Task Live_page_embeds_model_class_order()
    {
        // Browser maps logits by index, so it needs the model's class order, not dex order.
        var html = await _client.GetStringAsync("/Live");
        var expected = File.ReadLines(Path.Combine(AppContext.BaseDirectory, "Data", "best_sketch_classes.txt"))
            .Where(l => l.Trim().Length > 0).Take(3).Select(l => $"\"{l.Trim()}\"");
        Assert.Contains("window.LIVE_CLASSES = [" + string.Join(",", expected), html);
    }

    // Browsers holding the v2 (Gen 1-3) model must re-download: new URLs + new cache version.
    [Fact]
    public void Live_script_uses_v3_gen1_9_models()
    {
        var dir = AppContext.BaseDirectory;
        while (!Directory.Exists(Path.Combine(dir, "src", "PokeGuess.Web"))) dir = Path.GetDirectoryName(dir)!;
        var js = File.ReadAllText(Path.Combine(dir, "src", "PokeGuess.Web", "wwwroot", "js", "live", "live.js"));
        Assert.Contains("convnext_gen1-9.onnx", js);
        Assert.Contains("convnext_gen1-9_int8.onnx", js);
        Assert.Contains("MODEL_VERSION = \"v3\"", js);
        Assert.DoesNotContain("convnext_gen1-3", js);
    }

    [Fact]
    public async Task Nav_has_live_link() =>
        Assert.Contains("href=\"/Live\"", await _client.GetStringAsync("/"));

    [Fact]
    public async Task Home_does_not_load_onnx_runtime() =>
        Assert.DoesNotContain("onnxruntime-web", await _client.GetStringAsync("/"));
}
