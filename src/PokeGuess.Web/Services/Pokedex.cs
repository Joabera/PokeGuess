using System.Text.Json;

namespace PokeGuess.Web.Services;

/// The 386 Gen 1–3 names in National Dex order, plus their PokéAPI slugs.
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

    /// 1, 2 or 3 by dex number (#1–151, #152–251, #252–386); 0 if unknown.
    public int GenerationOf(string name) => _index.TryGetValue(name, out var i)
        ? i < 151 ? 1 : i < 251 ? 2 : 3
        : 0;
}
