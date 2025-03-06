# Guía para contribuir al proyecto

## Creación de branches
- Primero clona el repositorio con ssh,
 usando tu clave ssh, si no tienes creala y asociala con github,
 despues usando cd ingrersas al mech, una vez ahí para hacer cambios,
 primero crea un branch nuevo, 
 Usa `git checkout -b nombre-del-branch`,

##Cómo hacer un Pull Request
1. Crea un branch para tus cambios.
2. Haz commits con cambios de UN solo tema.
3. Sube los cambios con `git push origin nombre-del-branch`.
4. Ve a GitHub y crea un Pull Request contra `master`.

## Uso de Jupyter-BookB
- Para compilar el libro, usa `jupyter-book build`.
- Para ver los cambios, usa `jupyter-book serve`.

## Uso de Git en la terBminal
- Usa `git add -p` para agregar cambios selectivamente a staging.

## Despliegue automático del sitio
Cada vez que se hacen cambios en la rama `master`, el sitio web se despliega automáticamente.  
Para asegurarse de que los cambios se reflejen correctamente:
- Revisar que el contenido es correcto antes de fusionar con `master`.
- Esperar unos minutos después de hacer `merge` en `master` para ver los cambios en producción.

