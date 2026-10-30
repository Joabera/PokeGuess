using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace PokeGuess.Web.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddGuessSource : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "Source",
                table: "Guesses",
                type: "nvarchar(10)",
                maxLength: 10,
                nullable: false,
                defaultValue: "upload");
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Source",
                table: "Guesses");
        }
    }
}
