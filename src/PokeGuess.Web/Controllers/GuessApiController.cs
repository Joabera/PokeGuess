using System.Data.Common;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.RateLimiting;
using Microsoft.EntityFrameworkCore;
using PokeGuess.Web.Data;
using PokeGuess.Web.Services;

namespace PokeGuess.Web.Controllers;

[ApiController]
[Route("api/guess")]
public class GuessApiController(PokeGuessDb db, InferenceClient inference, ILogger<GuessApiController> log)
    : ControllerBase
{
    public const long MaxUploadBytes = 5 * 1024 * 1024;
    public const string BadImage = "Couldn't read that image.";
    public const string Unavailable = "The Pokédex is waking up, try again in a few seconds.";

    [HttpPost]
    [RequestSizeLimit(6 * 1024 * 1024)]
    [EnableRateLimiting("guess")]
    public async Task<IActionResult> Guess(IFormFile? file, CancellationToken ct)
    {
        if (file is null || file.Length == 0 || file.Length > MaxUploadBytes ||
            !file.ContentType.StartsWith("image/", StringComparison.OrdinalIgnoreCase))
            return BadRequest(new { error = BadImage });

        InferenceResult result;
        try
        {
            await using var stream = file.OpenReadStream();
            result = await inference.PredictAsync(stream, file.FileName, file.ContentType, ct);
        }
        catch (InferenceBadImageException)
        {
            return BadRequest(new { error = BadImage });
        }
        catch (InferenceUnavailableException e)
        {
            log.LogWarning(e, "Inference unavailable");
            return StatusCode(StatusCodes.Status503ServiceUnavailable, new { error = Unavailable });
        }

        int? guessId = null;
        try
        {
            var top = result.Predictions[0];
            var guess = new Guess
            {
                CreatedAt = DateTime.UtcNow,
                PredictedName = top.Name,
                Confidence = top.Confidence,
                Model = result.Model,
            };
            db.Guesses.Add(guess);
            await db.SaveChangesAsync(ct);
            guessId = guess.Id;
        }
        catch (Exception e) when (e is DbUpdateException or DbException)
        {
            // Still show the prediction; the UI hides feedback when guessId is null.
            log.LogError(e, "Failed to save guess");
        }

        return Ok(new { guessId, model = result.Model, predictions = result.Predictions });
    }
}
