using System.Net;
using System.Net.Http.Headers;
using Microsoft.Extensions.DependencyInjection;
using PokeGuess.Web.Data;
using PokeGuess.Web.Services;

namespace PokeGuess.Web.Tests;

public sealed class StatsTests : IDisposable
{
    private readonly PokeGuessFactory _factory = new();
    private readonly HttpClient _client;

    public StatsTests() => _client = _factory.CreateClient();
    public void Dispose() => _factory.Dispose();

    private void Seed()
    {
        Guess G(string name, string model, bool? correct, string? actual = null) => new()
        {
            CreatedAt = DateTime.UtcNow, PredictedName = name, Confidence = 90m, Model = model,
            IsCorrect = correct, ActualName = actual ?? (correct == true ? name : null),
        };
        using var db = _factory.CreateDb();
        db.Guesses.AddRange(
            G("Pikachu", "legacy-gen1", true), G("Pikachu", "legacy-gen1", true), G("Pikachu", "legacy-gen1", true),
            G("Pikachu", "gen1-3", false, "Raichu"),
            G("Treecko", "gen1-3", true),
            G("Mudkip", "gen1-3", null));
        db.SaveChanges();
    }

    private async Task<StatsDto> Stats()
    {
        using var scope = _factory.Services.CreateScope();
        return await scope.ServiceProvider.GetRequiredService<StatsService>().GetAsync(CancellationToken.None);
    }

    [Fact]
    public async Task Empty_db_has_null_accuracy()
    {
        var s = await Stats();
        Assert.Equal(0, s.TotalGuesses);
        Assert.Null(s.Accuracy);
        Assert.Equal(["Gen 1", "Gen 2", "Gen 3"], s.ByGeneration.Select(g => g.Key));
        Assert.All(s.ByGeneration, g => Assert.Null(g.Accuracy));
        Assert.Empty(s.ByModel);
        Assert.Equal(["upload", "live"], s.BySource.Select(g => g.Key));
        Assert.All(s.BySource, g => Assert.Null(g.Accuracy));
        Assert.Equal(HttpStatusCode.OK, (await _client.GetAsync("/api/stats")).StatusCode);
    }

    [Fact]
    public async Task Aggregates_seeded_rows()
    {
        Seed();
        var s = await Stats();

        Assert.Equal((6, 5, 80.0), (s.TotalGuesses, s.WithFeedback, s.Accuracy));

        var byModel = s.ByModel.ToDictionary(m => m.Key);
        Assert.Equal(["gen1-3", "legacy-gen1"], s.ByModel.Select(m => m.Key));
        Assert.Equal((2, 1, 50.0), (byModel["gen1-3"].WithFeedback, byModel["gen1-3"].Correct, byModel["gen1-3"].Accuracy));
        Assert.Equal((3, 3, 100.0), (byModel["legacy-gen1"].WithFeedback, byModel["legacy-gen1"].Correct, byModel["legacy-gen1"].Accuracy));

        var gens = s.ByGeneration;
        Assert.Equal((4, 3, 75.0), (gens[0].WithFeedback, gens[0].Correct, gens[0].Accuracy));
        Assert.Equal((0, 0, (double?)null), (gens[1].WithFeedback, gens[1].Correct, gens[1].Accuracy));
        Assert.Equal((1, 1, 100.0), (gens[2].WithFeedback, gens[2].Correct, gens[2].Accuracy));

        Assert.Equal(("Pikachu", 4), (s.MostGuessed[0].Name, s.MostGuessed[0].Count));
        Assert.Equal(["Pikachu", "Mudkip", "Treecko"], s.MostGuessed.Select(m => m.Name)); // ties by name
        var mix = Assert.Single(s.TopMixUps);
        Assert.Equal(("Pikachu", "Raichu", 1), (mix.Predicted, mix.Actual, mix.Count));
    }

    [Fact]
    public async Task BySource_splits_upload_and_live()
    {
        Seed(); // all "upload": 5 with feedback, 4 correct
        using (var db = _factory.CreateDb())
        {
            db.Guesses.AddRange(
                new Guess { CreatedAt = DateTime.UtcNow, PredictedName = "Pikachu", Confidence = 90m, Model = "gen1-3", Source = "live", IsCorrect = true, ActualName = "Pikachu" },
                new Guess { CreatedAt = DateTime.UtcNow, PredictedName = "Pikachu", Confidence = 90m, Model = "gen1-3", Source = "live", IsCorrect = false, ActualName = "Raichu" });
            db.SaveChanges();
        }
        var s = await Stats();
        Assert.Equal(["upload", "live"], s.BySource.Select(g => g.Key));
        Assert.Equal((5, 4, 80.0), (s.BySource[0].WithFeedback, s.BySource[0].Correct, s.BySource[0].Accuracy));
        Assert.Equal((2, 1, 50.0), (s.BySource[1].WithFeedback, s.BySource[1].Correct, s.BySource[1].Accuracy));
    }

    [Fact]
    public async Task Api_stats_returns_json_by_default()
    {
        Seed();
        var res = await _client.GetAsync("/api/stats");
        Assert.Equal("application/json", res.Content.Headers.ContentType!.MediaType);
        Assert.Contains("\"totalGuesses\":6", await res.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task Api_stats_returns_xml_when_asked()
    {
        Seed();
        var req = new HttpRequestMessage(HttpMethod.Get, "/api/stats");
        req.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/xml"));
        var res = await _client.SendAsync(req);
        Assert.Equal("application/xml", res.Content.Headers.ContentType!.MediaType);
        var xml = await res.Content.ReadAsStringAsync();
        Assert.Contains("<TotalGuesses>6</TotalGuesses>", xml);
        Assert.Contains("<BySource>", xml);
    }

    [Fact]
    public async Task Stats_page_renders()
    {
        async Task<string> Page() => WebUtility.HtmlDecode(await _client.GetStringAsync("/Stats"));
        Assert.Contains("—", await Page());
        Seed();
        Assert.Contains("80.0%", await Page());
    }
}
