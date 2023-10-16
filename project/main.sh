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
    basename_file=$(basename "$file")
    if [[ "${file}" == *"error_"* ]];
    then
        continue
    fi

    mime_type=$(file -b --mime-type "$file")
    echo "'${file} - ${mime_type}'"

    if [[ ${mime_type} = "application/pdf" ]]
    then
        python3 "${PATH_IDP_DOCKER_SCRIPTS}"/scripts_for_IDP/len_files_in_file.py "${pdf_path}/${basename_file}"
        rm -rf "${pdf_path}/cache" && mkdir "${pdf_path}/cache"
        python3 "${PATH_IDP_DOCKER_SCRIPTS}"/scripts_for_IDP/pdf_split.py "$file" "${pdf_path}/cache/${basename_file}"
        python3 "${PATH_IDP_DOCKER_SCRIPTS}"/scripts_for_IDP/len_files_in_cache.py "${pdf_path}/cache"
        python3 "${PATH_IDP_DOCKER_SCRIPTS}"/run_all_scripts.py "${pdf_path}" "${file}"
    elif [[ ${mime_type} = "image/jpeg" ]]
    then
        cp "${file}" "${pdf_path}/cache"
        python3 "${PATH_IDP_DOCKER_SCRIPTS}"/run_all_scripts.py "${pdf_path}" "${file}"
    else
        echo "ERROR: unsupported format ${mime_type}"
        mv "${file}" "${pdf_path}/error_$(basename "${file}")"
        continue
    fi

    mv "${file}" "${done_path}"
done
