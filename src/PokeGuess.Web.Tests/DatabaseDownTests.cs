using System.Net;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;

namespace PokeGuess.Web.Tests;

/// The free-tier DB can be paused for the rest of the month; the site must still start.
public sealed class DatabaseDownTests : IDisposable
{
    private readonly WebApplicationFactory<Program> _factory = new WebApplicationFactory<Program>()
        .WithWebHostBuilder(b =>
        {
            b.UseEnvironment("Testing");
            b.UseSetting("Database:Provider", "Sqlite");
            b.UseSetting("ConnectionStrings:Default", "Data Source=/no/such/dir/pokeguess.db");
            b.UseSetting("Inference:BaseUrl", "http://inference.test");
            b.UseSetting("Inference:ApiKey", "k");
        });

    public void Dispose() => _factory.Dispose();

    [Fact]
    public async Task App_starts_and_serves_home_when_db_is_unreachable()
    {
        var res = await _factory.CreateClient().GetAsync("/");
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
    }
}
