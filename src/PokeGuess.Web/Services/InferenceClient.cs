using System.Net;
using System.Net.Http.Headers;

namespace PokeGuess.Web.Services;

public record Prediction(string Name, decimal Confidence);
public record InferenceResult(string Model, IReadOnlyList<Prediction> Predictions);

/// The inference service could not read the image (its 400).
public class InferenceBadImageException : Exception;

/// The inference service is down, timed out, or failed.
public class InferenceUnavailableException(string message, Exception? inner = null) : Exception(message, inner);

/// Typed client for the Flask inference service (POST /predict).
public class InferenceClient(HttpClient http)
{
    public async Task<InferenceResult> PredictAsync(Stream image, string fileName, string contentType, CancellationToken ct)
    {
        var file = new StreamContent(image);
        file.Headers.ContentType = MediaTypeHeaderValue.Parse(contentType);
        using var form = new MultipartFormDataContent { { file, "file", fileName } };

        HttpResponseMessage res;
        try
        {
            res = await http.PostAsync("/predict", form, ct);
        }
        catch (HttpRequestException e)
        {
            throw new InferenceUnavailableException("Inference service unreachable", e);
        }
        catch (TaskCanceledException e) when (!ct.IsCancellationRequested)
        {
            throw new InferenceUnavailableException("Inference service timed out", e);
        }

        using (res)
        {
            if (res.StatusCode == HttpStatusCode.BadRequest) throw new InferenceBadImageException();
            if (!res.IsSuccessStatusCode)
                throw new InferenceUnavailableException($"Inference service returned {(int)res.StatusCode}");
            return await res.Content.ReadFromJsonAsync<InferenceResult>(ct)
                ?? throw new InferenceUnavailableException("Inference service returned no body");
        }
    }
}
