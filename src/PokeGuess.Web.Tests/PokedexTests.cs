using PokeGuess.Web.Services;

namespace PokeGuess.Web.Tests;

public class PokedexTests
{
    private static readonly Pokedex Dex = Pokedex.Load(
        Path.Combine(AppContext.BaseDirectory, "Data", "class_names_v2.txt"),
        Path.Combine(AppContext.BaseDirectory, "Data", "pokeapi_slugs.json"));

    [Fact]
    public void Load_reads_1025_names()
    {
        Assert.Equal(1025, Dex.Names.Count);
        Assert.Equal("Bulbasaur", Dex.Names[0]);
        Assert.Equal("Turtwig", Dex.Names[386]);
        Assert.Equal("Pecharunt", Dex.Names[1024]);
        Assert.Equal("mr-mime", Dex.Slugs["Mr. Mime"]);
        Assert.Equal("giratina-altered", Dex.Slugs["Giratina"]);
    }

    // Both sides of every generation boundary (last dex #: 151, 251, 386, 493, 649, 721, 809, 905, 1025).
    [Theory]
    [InlineData("Chikorita", 2)]
    [InlineData("Turtwig", 4)]
    [InlineData("Arceus", 4)]
    [InlineData("Victini", 5)]
    [InlineData("Genesect", 5)]
    [InlineData("Chespin", 6)]
    [InlineData("Volcanion", 6)]
    [InlineData("Rowlet", 7)]
    [InlineData("Melmetal", 7)]
    [InlineData("Grookey", 8)]
    [InlineData("Enamorus", 8)]
    [InlineData("Sprigatito", 9)]
    [InlineData("Pecharunt", 9)]
    public void GenerationOf_gen_boundaries(string name, int gen) =>
        Assert.Equal(gen, Dex.GenerationOf(name));

    [Theory]
    [InlineData("Mew", 1)]
    [InlineData("Chikorita", 2)]
    [InlineData("Celebi", 2)]
    [InlineData("Treecko", 3)]
    [InlineData("Deoxys", 3)]
    [InlineData("Agumon", 0)]
    public void GenerationOf_uses_dex_order(string name, int gen) =>
        Assert.Equal(gen, Dex.GenerationOf(name));

    [Theory]
    [InlineData(" pikachu ", "Pikachu")]
    [InlineData("mr. mime", "Mr. Mime")]
    public void TryCanonical_is_case_insensitive(string input, string expected)
    {
        Assert.True(Dex.TryCanonical(input, out var canonical));
        Assert.Equal(expected, canonical);
    }

    [Fact]
    public void TryCanonical_rejects_unknown() => Assert.False(Dex.TryCanonical("Agumon", out _));
}
