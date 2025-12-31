# FluxEM Demo
Output docs/demo.gif

Set FontSize 16
Set Width 800
Set Height 400
Set Theme "Dracula"

Type "python3"
Enter
Sleep 500ms

Type "from fluxem import create_unified_model"
Enter
Sleep 300ms

Type "model = create_unified_model()"
Enter
Sleep 300ms

Type "# Let's try some arithmetic"
Enter
Sleep 500ms

Type "model.compute('1847 * 392')"
Enter
Sleep 800ms

Type "model.compute('123456 + 789')"
Enter
Sleep 800ms

Type "model.compute('2 ** 16')"
Enter
Sleep 800ms

Type "model.compute('56088 / 123')"
Enter
Sleep 1s

Type "exit()"
Enter
