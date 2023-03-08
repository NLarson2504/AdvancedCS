extends TextureProgress

func _on_Control_healthChanged(health):
	self.value = health
	
