extends Control

signal healthChanged(health)
signal healthChangedTwo(health)

func _on_PlayerOne_updateP1BarValue(value):
	emit_signal("healthChanged", value)


func _on_PlayerTwo_updateP2BarValue(value):
	emit_signal("healthChangedTwo", value)
