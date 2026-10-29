namespace PokeGuess.Web.Data;

/// One prediction shown to a user, plus the feedback they gave on it (if any).
public class Guess
{
    public int Id { get; set; }
    public DateTime CreatedAt { get; set; }
    public required string PredictedName { get; set; }
    public decimal Confidence { get; set; }
    public required string Model { get; set; }
    public bool? IsCorrect { get; set; }
    public string? ActualName { get; set; }
}
