# FluxEM Demo
Output docs/demo.gif

Set FontSize 16
Set Width 800
Set Height 450
Set Theme "Dracula"

# Hide venv activation
Hide
Type "source .venv/bin/activate"
Enter
Sleep 500ms
Show

Type "python3"
Enter
Sleep 800ms

Type "from fluxem import create_unified_model"
Enter
Sleep 500ms

Type "model = create_unified_model()"
Enter
Sleep 500ms

Type "model.compute('1234 + 5678')"
Enter
Sleep 1s

Type "model.compute('250 * 4')"
Enter
Sleep 1s

Type "model.compute('1000 / 8')"
Enter
Sleep 1s

Type "model.compute('3 ** 4')"
Enter
Sleep 1.5s

Type "exit()"
Enter
Sleep 500ms
