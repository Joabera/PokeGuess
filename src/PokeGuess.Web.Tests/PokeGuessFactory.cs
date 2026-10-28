using System.Net;
using System.Text;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.TestHost;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using PokeGuess.Web.Data;

namespace PokeGuess.Web.Tests;

/// One instance per test: a private in-memory SQLite DB and fake inference service.
public sealed class PokeGuessFactory : WebApplicationFactory<Program>
{
    public const string ApiKey = "test-key";
    public SqliteConnection Connection { get; } = new("DataSource=:memory:");
    public FakeInferenceHandler Inference { get; } = new();

    public PokeGuessFactory() => Connection.Open();

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseEnvironment("Testing");
        builder.UseSetting("Database:Provider", "Sqlite");
        builder.UseSetting("Inference:BaseUrl", "http://inference.test");
        builder.UseSetting("Inference:ApiKey", ApiKey);
        builder.ConfigureTestServices(services =>
        {
            // EF Core composes AddDbContext configs, so drop the app's before adding ours.
            services.RemoveAll<IDbContextOptionsConfiguration<PokeGuessDb>>();
            services.AddDbContext<PokeGuessDb>(o => o.UseSqlite(Connection));
        });
    }

    public PokeGuessDb CreateDb() =>
        new(new DbContextOptionsBuilder<PokeGuessDb>().UseSqlite(Connection).Options);

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);
        if (disposing) Connection.Dispose();
    }
}

public sealed class FakeInferenceHandler : HttpMessageHandler
{
    public const string DefaultBody =
        """{"model":"gen1-3","predictions":[{"name":"Pikachu","confidence":97.31},{"name":"Raichu","confidence":1.5},{"name":"Pichu","confidence":0.4}]}""";

    public Func<HttpRequestMessage, HttpResponseMessage> Respond { get; set; } = _ => Json(HttpStatusCode.OK, DefaultBody);
    public HttpRequestMessage? LastRequest { get; private set; }

    public static HttpResponseMessage Json(HttpStatusCode status, string body) =>
        new(status) { Content = new StringContent(body, Encoding.UTF8, "application/json") };

    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct)
    {
        LastRequest = request;
        return Task.FromResult(Respond(request));
    }
}
