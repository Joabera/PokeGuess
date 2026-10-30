using Microsoft.AspNetCore.Mvc;

namespace PokeGuess.Web.Controllers;

public class LiveController : Controller
{
    // The in-browser model's output order (ImageFolder/alphabetical), not dex order.
    private static readonly Lazy<string[]> ModelClasses = new(() =>
        System.IO.File.ReadLines(Path.Combine(AppContext.BaseDirectory, "Data", "best_sketch_classes.txt"))
            .Select(l => l.Trim()).Where(l => l.Length > 0).ToArray());

    public IActionResult Index() => View(ModelClasses.Value);
}
