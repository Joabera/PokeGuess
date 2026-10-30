using System.Threading.RateLimiting;
using Microsoft.AspNetCore.HttpOverrides;
using Microsoft.AspNetCore.StaticFiles;
using Microsoft.Extensions.FileProviders;
using Microsoft.EntityFrameworkCore;
using PokeGuess.Web.Data;
using PokeGuess.Web.Services;

var builder = WebApplication.CreateBuilder(args);
var config = builder.Configuration;

builder.Services.AddControllersWithViews().AddXmlSerializerFormatters();
builder.Services.AddScoped<StatsService>();

// SQL Server (Azure SQL) in production; SQLite for local dev and tests.
var useSqlServer = config["Database:Provider"] == "SqlServer";
builder.Services.AddDbContext<PokeGuessDb>(o =>
{
    var cs = config.GetConnectionString("Default");
    // Retries cover the free serverless tier resuming from auto-pause (error 40613),
    // kept short so a guess still returns (with guessId null) if the DB stays down.
    if (useSqlServer)
        o.UseSqlServer(cs, sql => sql.EnableRetryOnFailure(3, TimeSpan.FromSeconds(10), null));
    else
        o.UseSqlite(cs);
});

builder.Services.AddSingleton(_ => Pokedex.Load(
    Path.Combine(AppContext.BaseDirectory, "Data", "class_names_v2.txt"),
    Path.Combine(AppContext.BaseDirectory, "Data", "pokeapi_slugs.json")));

builder.Services.AddHttpClient<InferenceClient>(http =>
{
    http.BaseAddress = new Uri(config["Inference:BaseUrl"]!);
    http.DefaultRequestHeaders.Add("X-Api-Key", config["Inference:ApiKey"]);
    http.Timeout = TimeSpan.FromSeconds(30); // allows for an inference cold start
});

// App Service's front end is the only proxy, so trust X-Forwarded-For from it.
builder.Services.Configure<ForwardedHeadersOptions>(o =>
{
    o.ForwardedHeaders = ForwardedHeaders.XForwardedFor | ForwardedHeaders.XForwardedProto;
    o.KnownIPNetworks.Clear();
    o.KnownProxies.Clear();
});

// Inference costs CPU we pay for: cap guesses per client IP.
builder.Services.AddRateLimiter(o =>
{
    o.AddPolicy("guess", http => RateLimitPartition.GetFixedWindowLimiter(
        http.Connection.RemoteIpAddress?.ToString() ?? "unknown",
        _ => new FixedWindowRateLimiterOptions
        {
            PermitLimit = config.GetValue("RateLimit:GuessPerMinute", 10),
            Window = TimeSpan.FromMinutes(1),
            QueueLimit = 0,
        }));
    o.OnRejected = (ctx, ct) =>
    {
        ctx.HttpContext.Response.StatusCode = StatusCodes.Status429TooManyRequests;
        return new ValueTask(ctx.HttpContext.Response.WriteAsJsonAsync(
            new { error = "Slow down, trainer, try again in a minute." }, ct));
    };
});

var app = builder.Build();

app.UseForwardedHeaders();

// Migrations are generated for SQL Server only; SQLite just creates the schema.
// Never fatal: a paused/unreachable DB must not stop guessing (feedback/stats degrade instead).
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<PokeGuessDb>();
    try
    {
        if (useSqlServer) db.Database.Migrate(); else db.Database.EnsureCreated();
    }
    catch (Exception e)
    {
        app.Logger.LogError(e, "Database schema setup failed; continuing without it");
    }
}

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseRouting();
app.UseRateLimiter();

// Live page models (~113 MB / 28 MB): plain static files outside wwwroot so the build
// doesn't fingerprint or compress them. Only .onnx is served. live.js requests them with
// ?v=<model version>, so a new release is a new URL and they can be cached forever.
var modelsDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, config["LiveModels:Path"] ?? "models"));
if (Directory.Exists(modelsDir))
{
    var onnxOnly = new FileExtensionContentTypeProvider(new Dictionary<string, string>
    {
        [".onnx"] = "application/octet-stream",
    });
    app.UseStaticFiles(new StaticFileOptions
    {
        FileProvider = new PhysicalFileProvider(modelsDir),
        RequestPath = "/models",
        ContentTypeProvider = onnxOnly,
        OnPrepareResponse = ctx => ctx.Context.Response.Headers.CacheControl = "public, max-age=31536000, immutable",
    });
}

app.MapStaticAssets();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}")
    .WithStaticAssets();

app.Run();

public partial class Program;
