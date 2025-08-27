.PHONY: get-raw-data

get-raw-data:
	curl "https://api.dotgg.gg/cgfw/getcards?game=pokepocket" > raw_data.json
	cat raw_data.json | jq > pokepocket.json
	rm raw_data.json
