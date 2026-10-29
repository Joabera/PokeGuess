using Microsoft.EntityFrameworkCore;

namespace PokeGuess.Web.Data;

public class PokeGuessDb(DbContextOptions<PokeGuessDb> options) : DbContext(options)
{
    public DbSet<Guess> Guesses => Set<Guess>();

    protected override void OnModelCreating(ModelBuilder b)
    {
        var guess = b.Entity<Guess>();
        guess.Property(g => g.PredictedName).HasMaxLength(40);
        guess.Property(g => g.ActualName).HasMaxLength(40);
        guess.Property(g => g.Model).HasMaxLength(20);
        guess.Property(g => g.Confidence).HasPrecision(5, 2);
    }
}
