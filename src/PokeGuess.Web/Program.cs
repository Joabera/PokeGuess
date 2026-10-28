using Microsoft.EntityFrameworkCore;
using PokeGuess.Web.Data;
using PokeGuess.Web.Services;

var builder = WebApplication.CreateBuilder(args);
var config = builder.Configuration;

builder.Services.AddControllersWithViews();

// SQL Server (Azure SQL) in production; SQLite for local dev and tests.
var useSqlServer = config["Database:Provider"] == "SqlServer";
builder.Services.AddDbContext<PokeGuessDb>(o =>
{
    var cs = config.GetConnectionString("Default");
    if (useSqlServer) o.UseSqlServer(cs); else o.UseSqlite(cs);
});

builder.Services.AddSingleton(_ => Pokedex.Load(
    Path.Combine(AppContext.BaseDirectory, "Data", "class_names_v2.txt"),
    Path.Combine(AppContext.BaseDirectory, "Data", "pokeapi_slugs.json")));

var app = builder.Build();

// Migrations are generated for SQL Server only; SQLite just creates the schema.
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<PokeGuessDb>();
    if (useSqlServer) db.Database.Migrate(); else db.Database.EnsureCreated();
}

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseRouting();

app.MapStaticAssets();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}")
    .WithStaticAssets();

app.Run();

public partial class Program;
