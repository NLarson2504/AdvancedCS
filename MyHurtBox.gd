class_name MyHurtBox
extends Area2D
signal hitBoxHurtBoxCollision(value, damage)
signal exited(value)

func _init() -> void:
	collision_layer = 0
	collision_mask = 2
	
func _ready() -> void:
	connect("area_entered", self, "_on_area_entered")
	connect("area_exited", self, "_on_area_exited")
	
func _on_area_entered(hitbox: MyHitBox) -> void:
	if hitbox == null:
		return
	emit_signal("hitBoxHurtBoxCollision", true, hitbox.damage)
	
func _on_area_exited(hitbox: MyHitBox) -> void:
	if hitbox == null:
		return
	emit_signal("exited", false)
