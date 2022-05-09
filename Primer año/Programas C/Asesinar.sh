#!/bin/bash
# Este programa lo voy a usar para matar todos los programas que haya mandado a correr juntos.

head -1 salida* | cut -f 6 -d ' ' | kill



