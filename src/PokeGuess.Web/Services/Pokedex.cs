using System.Text.Json;

namespace PokeGuess.Web.Services;

/// The 1025 Gen 1–9 names in National Dex order, plus their PokéAPI slugs.
public sealed class Pokedex
{
    private readonly Dictionary<string, int> _index;

    public Pokedex(IReadOnlyList<string> dexOrderNames, IReadOnlyDictionary<string, string> slugs)
    {
        Names = dexOrderNames;
        Slugs = slugs;
        _index = dexOrderNames
            .Select((name, i) => (name, i))
            .ToDictionary(x => x.name, x => x.i, StringComparer.OrdinalIgnoreCase);
    }

    public IReadOnlyList<string> Names { get; }
    public IReadOnlyDictionary<string, string> Slugs { get; }

    public static Pokedex Load(string namesPath, string slugsPath) => new(
        File.ReadLines(namesPath).Select(l => l.Trim()).Where(l => l.Length > 0).ToList(),
        JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(slugsPath))!);

    public bool TryCanonical(string? name, out string canonical)
    {
        if (name is not null && _index.TryGetValue(name.Trim(), out var i))
        {
            canonical = Names[i];
            return true;
        }
        canonical = "";
        return false;
    }

    // Last National Dex number of each generation, Gen 1..9.
    private static readonly int[] GenerationEnds = [151, 251, 386, 493, 649, 721, 809, 905, 1025];

    /// 1–9 by dex number; 0 if unknown.
    public int GenerationOf(string name) => _index.TryGetValue(name, out var i)
        ? Array.FindIndex(GenerationEnds, end => i < end) + 1
        : 0;
}
