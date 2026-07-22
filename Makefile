
all: prepare
	echo DONE

prepare:
	@echo "Before running export CDSAPI_KEY and CDSAPI_URL environment variables."
	@mkdir --parents copernicus
	uv run copernicus_helper\
		--variable total_precipitation\
		--country CN\
		--dataset land\
		--time-range 2019-2020\
		--folder "copernicus"
