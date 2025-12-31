# FluxEM Demo
Output docs/demo.gif

Set FontSize 16
Set Width 800
Set Height 450
Set Theme "Dracula"

Type "source .venv/bin/activate"
Enter
Sleep 500ms

Type "python3"
Enter
Sleep 800ms

Type "from fluxem import create_unified_model"
Enter
Sleep 500ms

Type "model = create_unified_model()"
Enter
Sleep 500ms

Type "model.compute('1847 * 392')"
Enter
Sleep 1s

Type "model.compute('123456 + 789')"
Enter
Sleep 1s

Type "model.compute('2 ** 16')"
Enter
Sleep 1s

Type "model.compute('56088 / 123')"
Enter
Sleep 1.5s

Type "exit()"
Enter
Sleep 500ms
