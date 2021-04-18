set(BLOSSOM_DATA_DIR "${CMAKE_INSTALL_PREFIX}/share/blossom" CACHE INTERNAL "BLOSSOM_DATA_DIR")

install(FILES
	fa-regular-400.ttf
	fa-solid-900.ttf
	SourceSansPro-Regular.ttf
	DESTINATION ${BLOSSOM_DATA_DIR}
)
