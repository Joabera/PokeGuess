using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.EntityFrameworkCore;

namespace PokeGuess.Web.Tests;

public sealed class GuessApiTests : IDisposable
{
    private readonly PokeGuessFactory _factory = new();
    private readonly HttpClient _client;

    public GuessApiTests() => _client = _factory.CreateClient();
    public void Dispose() => _factory.Dispose();

    internal static MultipartFormDataContent Upload(int bytes = 128, string contentType = "image/png")
    {
        var file = new ByteArrayContent(new byte[bytes]);
        file.Headers.ContentType = new MediaTypeHeaderValue(contentType);
        return new MultipartFormDataContent { { file, "file", "x.png" } };
    }

    private Task<HttpResponseMessage> PostGuess(MultipartFormDataContent? body = null, string? forwardedFor = null)
    {
        var req = new HttpRequestMessage(HttpMethod.Post, "/api/guess") { Content = body ?? Upload() };
        if (forwardedFor is not null) req.Headers.Add("X-Forwarded-For", forwardedFor);
        return _client.SendAsync(req);
    }

    private static async Task<string?> Error(HttpResponseMessage res) =>
        (await res.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("error").GetString();

    [Fact]
    public async Task Guess_happy_path_saves_row()
    {
        var res = await PostGuess();
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);

        var body = await res.Content.ReadFromJsonAsync<JsonElement>();
        var id = body.GetProperty("guessId").GetInt32();
        Assert.True(id > 0);
        Assert.Equal("gen1-3", body.GetProperty("model").GetString());
        Assert.Equal("Pikachu", body.GetProperty("predictions")[0].GetProperty("name").GetString());
        Assert.Equal(97.31m, body.GetProperty("predictions")[0].GetProperty("confidence").GetDecimal());

        using var db = _factory.CreateDb();
        var row = await db.Guesses.SingleAsync();
        Assert.Equal(("Pikachu", 97.31m, "gen1-3"), (row.PredictedName, row.Confidence, row.Model));
        Assert.Null(row.IsCorrect);
        Assert.Null(row.ActualName);
    }

    [Fact]
    public async Task Guess_forwards_api_key_and_file()
    {
        await PostGuess();
        var sent = _factory.Inference.LastRequest!;
        Assert.Equal(PokeGuessFactory.ApiKey, sent.Headers.GetValues("X-Api-Key").Single());
        Assert.Equal("/predict", sent.RequestUri!.AbsolutePath);
        Assert.IsType<MultipartFormDataContent>(sent.Content);
    }

    [Fact]
    public async Task Guess_rejects_oversize()
    {
        var res = await PostGuess(Upload(5 * 1024 * 1024 + 1));
        Assert.Equal(HttpStatusCode.BadRequest, res.StatusCode);
        Assert.Null(_factory.Inference.LastRequest);
    }

    [Fact]
    public async Task Guess_accepts_exactly_5_mb()
    {
        var res = await PostGuess(Upload(5 * 1024 * 1024));
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
    }

    [Fact]
    public async Task Guess_rejects_non_image_type()
    {
        var res = await PostGuess(Upload(contentType: "text/plain"));
        Assert.Equal(HttpStatusCode.BadRequest, res.StatusCode);
        Assert.Equal("Couldn't read that image.", await Error(res));
        Assert.Null(_factory.Inference.LastRequest);
    }

    [Fact]
    public async Task Guess_without_file_is_400()
    {
        var res = await PostGuess(new MultipartFormDataContent { { new StringContent("x"), "other" } });
        Assert.Equal(HttpStatusCode.BadRequest, res.StatusCode);
    }

    [Fact]
    public async Task Guess_inference_400_maps_to_400()
    {
        _factory.Inference.Respond = _ =>
            FakeInferenceHandler.Json(HttpStatusCode.BadRequest, """{"error":"Could not read image file."}""");
        var res = await PostGuess();
        Assert.Equal(HttpStatusCode.BadRequest, res.StatusCode);
        Assert.Equal("Couldn't read that image.", await Error(res));
    }

    [Theory]
    [InlineData("down")]
    [InlineData("500")]
    [InlineData("timeout")]
    public async Task Guess_inference_unavailable_maps_to_503(string failure)
    {
        _factory.Inference.Respond = failure switch
        {
            "down" => _ => throw new HttpRequestException("connection refused"),
            "timeout" => _ => throw new TaskCanceledException("timed out"),
            _ => _ => FakeInferenceHandler.Json(HttpStatusCode.InternalServerError, "{}"),
        };
        var res = await PostGuess();
        Assert.Equal(HttpStatusCode.ServiceUnavailable, res.StatusCode);
        Assert.Equal("The Pokédex is waking up, try again in a few seconds.", await Error(res));
    }

    [Fact]
    public async Task Guess_db_failure_still_returns_predictions()
    {
        _ = _factory.Services; // boot so the schema exists, then break it
        using (var cmd = _factory.Connection.CreateCommand())
        {
            cmd.CommandText = "DROP TABLE Guesses";
            cmd.ExecuteNonQuery();
        }

        var res = await PostGuess();
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        var body = await res.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal(JsonValueKind.Null, body.GetProperty("guessId").ValueKind);
        Assert.Equal("Pikachu", body.GetProperty("predictions")[0].GetProperty("name").GetString());
    }

    [Fact]
    public async Task Guess_rate_limited_per_forwarded_ip()
    {
        for (var i = 0; i < 10; i++)
            Assert.Equal(HttpStatusCode.OK, (await PostGuess(forwardedFor: "203.0.113.1")).StatusCode);

        var limited = await PostGuess(forwardedFor: "203.0.113.1");
        Assert.Equal(HttpStatusCode.TooManyRequests, limited.StatusCode);
        Assert.Equal("Slow down, trainer, try again in a minute.", await Error(limited));

        Assert.Equal(HttpStatusCode.OK, (await PostGuess(forwardedFor: "203.0.113.2")).StatusCode);
    }
}
