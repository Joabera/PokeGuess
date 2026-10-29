using Microsoft.AspNetCore.Mvc;
using PokeGuess.Web.Services;

namespace PokeGuess.Web.Controllers;

public class StatsController(StatsService stats) : Controller
{
    public async Task<IActionResult> Index(CancellationToken ct) => View(await stats.GetAsync(ct));

    /// Same numbers as the page; JSON by default, XML for Accept: application/xml.
    [HttpGet("/api/stats")]
    public async Task<IActionResult> Api(CancellationToken ct) => Ok(await stats.GetAsync(ct));
}
