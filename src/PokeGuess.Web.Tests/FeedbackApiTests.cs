using System.Net;
using System.Net.Http.Json;
using Microsoft.EntityFrameworkCore;
using PokeGuess.Web.Data;

namespace PokeGuess.Web.Tests;

public sealed class FeedbackApiTests : IDisposable
{
    private readonly PokeGuessFactory _factory = new();
    private readonly HttpClient _client;
    private readonly int _id;

    public FeedbackApiTests()
    {
        _client = _factory.CreateClient();
        using var db = _factory.CreateDb();
        var guess = new Guess { CreatedAt = DateTime.UtcNow, PredictedName = "Pikachu", Confidence = 90m, Model = "gen1-3" };
        db.Guesses.Add(guess);
        db.SaveChanges();
        _id = guess.Id;
    }

    public void Dispose() => _factory.Dispose();

    private Task<HttpResponseMessage> Send(object body, int? id = null) =>
        _client.PostAsJsonAsync($"/api/guess/{id ?? _id}/feedback", body);

    private Guess Row()
    {
        using var db = _factory.CreateDb();
        return db.Guesses.AsNoTracking().Single(g => g.Id == _id);
    }

    [Fact]
    public async Task Correct_sets_actual_to_predicted()
    {
        Assert.Equal(HttpStatusCode.NoContent, (await Send(new { isCorrect = true })).StatusCode);
        Assert.Equal((true, "Pikachu"), (Row().IsCorrect, Row().ActualName));
    }

    [Fact]
    public async Task Wrong_with_valid_name_canonicalizes()
    {
        Assert.Equal(HttpStatusCode.NoContent, (await Send(new { isCorrect = false, actualName = " raichu " })).StatusCode);
        Assert.Equal((false, "Raichu"), (Row().IsCorrect, Row().ActualName));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("Agumon")]
    public async Task Wrong_with_missing_or_unknown_name_is_400(string? name)
    {
        var res = await Send(new { isCorrect = false, actualName = name });
        Assert.Equal(HttpStatusCode.BadRequest, res.StatusCode);
        Assert.Contains("Pick a Pokémon from the list.", await res.Content.ReadAsStringAsync());
        Assert.Null(Row().IsCorrect);
    }

    [Fact]
    public async Task Unknown_id_is_404() =>
        Assert.Equal(HttpStatusCode.NotFound, (await Send(new { isCorrect = true }, id: 9999)).StatusCode);

    [Fact]
    public async Task Second_feedback_is_409()
    {
        Assert.Equal(HttpStatusCode.NoContent, (await Send(new { isCorrect = true })).StatusCode);
        var second = await Send(new { isCorrect = false, actualName = "Raichu" });
        Assert.Equal(HttpStatusCode.Conflict, second.StatusCode);
        Assert.Equal((true, "Pikachu"), (Row().IsCorrect, Row().ActualName));
    }
}
