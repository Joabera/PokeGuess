using System.Net;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;

namespace PokeGuess.Web.Tests;

/// /models/ serves the Live page's ONNX files and nothing else.
public sealed class ModelFilesTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "pokeguess-models-" + Guid.NewGuid().ToString("N"));
    private readonly PokeGuessFactory _base = new();
    private readonly WebApplicationFactory<Program> _factory;
    private readonly HttpClient _client;

    public ModelFilesTests()
    {
        var models = Path.Combine(_root, "models");
        Directory.CreateDirectory(models);
        File.WriteAllBytes(Path.Combine(models, "tiny.onnx"), new byte[16]);
        File.WriteAllText(Path.Combine(models, "secret.txt"), "not a model");
        File.WriteAllBytes(Path.Combine(_root, "outside.onnx"), new byte[8]);
        _factory = _base.WithWebHostBuilder(b => b.UseSetting("LiveModels:Path", models));
        _client = _factory.CreateClient();
    }

    public void Dispose()
    {
        _factory.Dispose();
        _base.Dispose();
        Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public async Task Onnx_file_is_served_with_long_cache()
    {
        var res = await _client.GetAsync("/models/tiny.onnx");
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        Assert.Equal("application/octet-stream", res.Content.Headers.ContentType!.MediaType);
        Assert.Equal(16, res.Content.Headers.ContentLength);
        var cache = res.Headers.CacheControl!.ToString();
        Assert.Contains("immutable", cache);
        Assert.Contains("max-age=31536000", cache);
    }

    [Fact]
    public async Task Non_onnx_file_is_404() =>
        Assert.Equal(HttpStatusCode.NotFound, (await _client.GetAsync("/models/secret.txt")).StatusCode);

    [Fact]
    public async Task Missing_file_is_404() =>
        Assert.Equal(HttpStatusCode.NotFound, (await _client.GetAsync("/models/nope.onnx")).StatusCode);

    [Theory]
    [InlineData("/models/..%2Foutside.onnx")]
    [InlineData("/models/%2E%2E/outside.onnx")]
    [InlineData("/models/../outside.onnx")]
    [InlineData("/models/..%5Coutside.onnx")]
    public async Task Traversal_is_404(string path)
    {
        var req = new HttpRequestMessage(HttpMethod.Get, new Uri("http://localhost" + path, UriKind.Absolute));
        Assert.Equal(HttpStatusCode.NotFound, (await _client.SendAsync(req)).StatusCode);
    }
}
