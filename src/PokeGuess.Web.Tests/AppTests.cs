namespace PokeGuess.Web.Tests;

public sealed class AppTests : IDisposable
{
    private readonly PokeGuessFactory _factory = new();
    public void Dispose() => _factory.Dispose();

    [Fact]
    public async Task App_starts_and_serves_home()
    {
        var res = await _factory.CreateClient().GetAsync("/");
        Assert.Equal(System.Net.HttpStatusCode.OK, res.StatusCode);
    }

    [Fact]
    public void Database_is_created_on_startup()
    {
        _ = _factory.Services; // boots the app
        using var db = _factory.CreateDb();
        Assert.Empty(db.Guesses);
    }
}
