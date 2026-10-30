using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.EntityFrameworkCore;

namespace PokeGuess.Web.Tests;

public sealed class SourceTests : IDisposable
{
    private readonly PokeGuessFactory _factory = new();
    private readonly HttpClient _client;

    public SourceTests() => _client = _factory.CreateClient();
    public void Dispose() => _factory.Dispose();

    private Task<HttpResponseMessage> PostGuess(string? source)
    {
        var form = GuessApiTests.Upload();
        if (source is not null) form.Add(new StringContent(source), "source");
        return _client.PostAsync("/api/guess", form);
    }

    [Fact]
    public async Task Live_source_is_saved()
    {
        Assert.Equal(HttpStatusCode.OK, (await PostGuess("live")).StatusCode);
        using var db = _factory.CreateDb();
        Assert.Equal("live", (await db.Guesses.SingleAsync()).Source);
    }

    [Fact]
    public async Task Missing_source_defaults_to_upload()
    {
        Assert.Equal(HttpStatusCode.OK, (await PostGuess(null)).StatusCode);
        using var db = _factory.CreateDb();
        Assert.Equal("upload", (await db.Guesses.SingleAsync()).Source);
    }

    [Fact]
    public async Task Unknown_source_is_400()
    {
        var res = await PostGuess("bogus");
        Assert.Equal(HttpStatusCode.BadRequest, res.StatusCode);
        var body = await res.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal("Unknown source.", body.GetProperty("error").GetString());
        Assert.Null(_factory.Inference.LastRequest);
        using var db = _factory.CreateDb();
        Assert.Empty(db.Guesses);
    }
}
