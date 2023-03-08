class_name SaveGame
extends Resource

const SAVE_GAME_PATH:= "user://savegame.tres"

export var jumpLocs := []

func write_savegame() -> void:
	ResourceSaver.save(SAVE_GAME_PATH, self)
	
static func load_savegame() -> Resource:
	if ResourceLoader.exists(SAVE_GAME_PATH):
		return load(SAVE_GAME_PATH)
	return null
