extends KinematicBody2D

const UP = Vector2(0,-1)
const GRAVITY = 20
const MAXFALLSPEED = 200
const MAXSPEED = 125
const JUMPFORCE = 370
const ACCEL = 22

var playerHealth = 100
onready var PlayerOne: AnimatedSprite = $SpriteOne

signal updateP1BarValue(value)
signal updateP2BarValue(value)

var facing_right = true
var can_idle = true
var in_double = false
var can_double = true
var moved = false
var motion = Vector2()
var changedBox = false
var hurtBoxEntered = false
var nextJumpCords = 0
var jumpLocs = []
var prev = 0
var limitEase = 2
var dubEase = 10
var posspos

func _ready():
	prev = loaded()
	print("LOADED: ", prev)
	jumpLocs = loaded().split(",", true)
	print("JUMPLOCS: ", jumpLocs)
	
func _physics_process(delta):
	
	motion.y += GRAVITY
	if motion.y > MAXFALLSPEED:
		motion.y = MAXFALLSPEED
		
	if facing_right == true:
		PlayerOne.scale.x = 1
		if changedBox == true:
			PlayerOne.position += Vector2(22, 0)
			changedBox = false
	else:
		PlayerOne.scale.x = -1
		if changedBox == false:
			PlayerOne.position -= Vector2(22, 0)
			changedBox = true
		
		
	motion.x = clamp(motion.x, -MAXSPEED, MAXSPEED)
	
	if can_idle:
		for number in range(limitEase):
			posspos = floor(PlayerOne.global_position.x)+number
			for location in jumpLocs:
				if (int(posspos) == int(location) && is_on_floor()):
					print("LOCATION HIT: ", location)
					if (("*" in location) && (can_double)):
						print("SUIII")
						motion.y = -JUMPFORCE*1.3
						can_double = false
					else:
						motion.y = -JUMPFORCE
		facing_right = true
		motion.x += ACCEL
		PlayerOne.play("run")
		if Input.is_action_pressed("left"):
			motion.x -= ACCEL
			facing_right = false
			if in_double == false && hurtBoxEntered == false:
				PlayerOne.play("run")
		elif hurtBoxEntered:
			PlayerOne.play("hurt")
		else:
			motion.x = lerp(motion.x, 0, 0.2)
			if !in_double && hurtBoxEntered == false:
				PlayerOne.play("idle")
			
		if is_on_floor():
			in_double = false
			can_double = true
			moved = false
			nextJumpCords = floor(PlayerOne.global_position.x - 5)
			if Input.is_action_just_pressed("jump"):
				motion.y = -JUMPFORCE
			
	if !is_on_floor():
		if in_double == false:
			PlayerOne.play("jump")
		if Input.is_action_just_pressed("jump"):
			if can_double:
				if !moved:
					motion.y = -JUMPFORCE
					moved = true
				in_double = true
				PlayerOne.play("doubleJump")
				yield(PlayerOne, "animation_finished")
				in_double = false
				can_double = false
		
	if Input.is_action_just_pressed("restart"):
		playerHealth = 100
		print("")
		get_tree().reload_current_scene()
		
	motion = move_and_slide(motion, UP)

func take_damage(amount: int) -> void:
	PlayerOne.play("hurt")


func _on_MyHurtBox_hitBoxHurtBoxCollision(value, damage) -> void:
	hurtBoxEntered = value
	playerHealth -= damage
	emit_signal("updateP1BarValue", playerHealth)
	if playerHealth == 0:
		print("LAST X: ", nextJumpCords)
		save(nextJumpCords)
		can_idle = false
		in_double = true
		motion.x = 0
		motion.y = 0
		PlayerOne.play("death")
		playerHealth = 100
		print("")
		get_tree().reload_current_scene()


func _on_MyHurtBox_exited(value):
	hurtBoxEntered = value
	
func save(nextJumpPos):
	var temp = ""
	var file = File.new()
	file.open("user://save_game.dat", File.WRITE)
	if (nextJumpCords > int(jumpLocs[0])):
		temp = str(nextJumpPos) + "," + prev
	else:
		temp = prev
	print("JUMPLOCS: ", jumpLocs[0])
	print("NEXTJUMPCORD: ", nextJumpCords)
	if (nextJumpCords <= int(jumpLocs[0])+dubEase && nextJumpCords >= int(jumpLocs[0])-dubEase) && not("*" in jumpLocs[0]):
		temp = "*" + temp
	file.store_string(temp)
	file.close()

func loaded():
	var file = File.new()
	file.open("user://save_game.dat", File.READ)
	var content = file.get_as_text()
	file.close()
	return content
