using PokeGuess.Web.Services;

namespace PokeGuess.Web.Tests;

public class PokedexTests
{
    private static readonly Pokedex Dex = Pokedex.Load(
        Path.Combine(AppContext.BaseDirectory, "Data", "class_names_v2.txt"),
        Path.Combine(AppContext.BaseDirectory, "Data", "pokeapi_slugs.json"));

    [Fact]
    public void Load_reads_386_names()
    {
        Assert.Equal(386, Dex.Names.Count);
        Assert.Equal("Bulbasaur", Dex.Names[0]);
        Assert.Equal("mr-mime", Dex.Slugs["Mr. Mime"]);
    }

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
