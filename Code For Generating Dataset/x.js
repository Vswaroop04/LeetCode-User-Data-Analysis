for (let i = 1; i <= 14000; i++) {
  const cell = worksheet['B' + i]
  if (cell && cell.v) {
    console.log(cell.v)

    const query = {
      query: `
        query languageStats($username: String!) {
          matchedUser(username: $username) {
            languageProblemCount {
              languageName
              problemsSolved
            }
          }
        }
      `,
      variables: {
        // Here cell.v gives every username in the dataset of users
        username: `${cell.v}`,
      },
      operationName: 'languageStats',
    }

    const response = await axios.post('https://leetcode.com/graphql/', query)

    const Data = {
      sno: val,
      username: cell.v,
      response,
    }

    const newWorkbook = XLSX.utils.book_new()
    const newSheet = XLSX.utils.json_to_sheet(Data)
    XLSX.utils.book_append_sheet(newWorkbook, newSheet, Data)

    XLSX.writeFile(
      newWorkbook,
      'D://Clg/SEM6/CS312/Project/Output/Dataset.xlsx',
    )
  }
}
