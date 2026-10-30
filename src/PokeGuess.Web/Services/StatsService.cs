using Microsoft.EntityFrameworkCore;
using PokeGuess.Web.Data;

namespace PokeGuess.Web.Services;

// Plain get/set classes (not records) so XmlSerializer can write them for /api/stats.
public class StatsDto
{
    public int TotalGuesses { get; set; }
    public int WithFeedback { get; set; }
    public double? Accuracy { get; set; }
    public List<GroupAccuracy> ByModel { get; set; } = [];
    public List<GroupAccuracy> ByGeneration { get; set; } = [];
    public List<GroupAccuracy> BySource { get; set; } = [];
    public List<NameCount> MostGuessed { get; set; } = [];
    public List<MixUp> TopMixUps { get; set; } = [];
}

public class GroupAccuracy
{
    public string Key { get; set; } = "";
    public int WithFeedback { get; set; }
    public int Correct { get; set; }
    public double? Accuracy { get; set; }
}

public class NameCount
{
    public string Name { get; set; } = "";
    public int Count { get; set; }
}

public class MixUp
{
    public string Predicted { get; set; } = "";
    public string Actual { get; set; } = "";
    public int Count { get; set; }
}

public class StatsService(PokeGuessDb db, Pokedex dex)
{
    /// Percent rounded to 1 decimal; null when there is no feedback to judge.
    private static double? Accuracy(int correct, int withFeedback) =>
        withFeedback == 0 ? null : Math.Round(100.0 * correct / withFeedback, 1);

    private static GroupAccuracy Group(string key, int withFeedback, int correct) =>
        new() { Key = key, WithFeedback = withFeedback, Correct = correct, Accuracy = Accuracy(correct, withFeedback) };

    public async Task<StatsDto> GetAsync(CancellationToken ct)
    {
        // At most 386 rows: one per predicted Pokémon. Totals and generations fold from this.
        var byName = await db.Guesses
            .GroupBy(g => g.PredictedName)
            .Select(g => new
            {
                Name = g.Key,
                Total = g.Count(),
                WithFeedback = g.Count(x => x.IsCorrect != null),
                Correct = g.Count(x => x.IsCorrect == true),
            })
            .ToListAsync(ct);

        var byModel = await db.Guesses
            .GroupBy(g => g.Model)
            .Select(g => new
            {
                Model = g.Key,
                WithFeedback = g.Count(x => x.IsCorrect != null),
                Correct = g.Count(x => x.IsCorrect == true),
            })
            .ToListAsync(ct);

        var bySource = await db.Guesses
            .GroupBy(g => g.Source)
            .Select(g => new
            {
                Source = g.Key,
                WithFeedback = g.Count(x => x.IsCorrect != null),
                Correct = g.Count(x => x.IsCorrect == true),
            })
            .ToListAsync(ct);

        var mixUps = await db.Guesses
            .Where(g => g.IsCorrect == false)
            .GroupBy(g => new { g.PredictedName, g.ActualName })
            .Select(g => new MixUp { Predicted = g.Key.PredictedName, Actual = g.Key.ActualName!, Count = g.Count() })
            .OrderByDescending(m => m.Count).ThenBy(m => m.Predicted).ThenBy(m => m.Actual)
            .Take(10)
            .ToListAsync(ct);

        int withFeedback = byName.Sum(n => n.WithFeedback), correct = byName.Sum(n => n.Correct);
        return new StatsDto
        {
            TotalGuesses = byName.Sum(n => n.Total),
            WithFeedback = withFeedback,
            Accuracy = Accuracy(correct, withFeedback),
            ByModel = byModel.OrderBy(m => m.Model, StringComparer.Ordinal)
                .Select(m => Group(m.Model, m.WithFeedback, m.Correct)).ToList(),
            ByGeneration = [.. new[] { 1, 2, 3 }.Select(gen =>
            {
                var rows = byName.Where(n => dex.GenerationOf(n.Name) == gen).ToList();
                return Group($"Gen {gen}", rows.Sum(n => n.WithFeedback), rows.Sum(n => n.Correct));
            })],
            BySource = [.. new[] { "upload", "live" }.Select(source =>
            {
                var row = bySource.FirstOrDefault(s => s.Source == source);
                return Group(source, row?.WithFeedback ?? 0, row?.Correct ?? 0);
            })],
            MostGuessed = byName.OrderByDescending(n => n.Total).ThenBy(n => n.Name, StringComparer.Ordinal)
                .Take(10).Select(n => new NameCount { Name = n.Name, Count = n.Total }).ToList(),
            TopMixUps = mixUps,
        };
    }
}
