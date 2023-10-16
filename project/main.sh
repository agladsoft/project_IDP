#!/bin/bash

export OMP_THREAD_LIMIT=1

pdf_path=${PATH_IDP_DOCKER_FILES}/files

cache="${pdf_path}"/cache
if [ ! -d "$cache" ]; then
  mkdir "${cache}"
fi

csv_path="${pdf_path}"/csv
if [ ! -d "$csv_path" ]; then
  mkdir "${csv_path}"
fi

json_path="${pdf_path}"/json
if [ ! -d "$json_path" ]; then
  mkdir "${json_path}"
fi

txt_path="${pdf_path}"/txt
if [ ! -d "$txt_path" ]; then
  mkdir "${txt_path}"
fi

done_path="${pdf_path}"/done
if [ ! -d "$done_path" ]; then
  mkdir "${done_path}"
fi


# shellcheck disable=SC2162
find "${pdf_path}" -maxdepth 1 -type f \( -name "*.pdf*" -or -name "*.jpg*" \) ! -newermt '3 seconds ago' -print0 | while read -d $'\0' file
do
    if [[ "${file}" == *"error_"* ]];
    then
        continue
    fi

    python3 "${PATH_IDP_DOCKER_SCRIPTS}"/main.py "${file}" "${cache}"

    # shellcheck disable=SC2181
    if [ $? -eq 0 ]
    then
        mv "${file}" "${done_path}"
    else
        mv "${file}" "${pdf_path}/error_$(basename "${file}")"
    fi
done
