const express = require('express')
const app = express()
const axios = require('axios')
const XLSX = require('xlsx')

app.use(express.json())
// const workbook = XLSX.readFile(
//   'D://Clg/SEM6/CS312/Project/leetcode_indian_userrating.csv',
// )

// const sheetName = workbook.SheetNames[0]

// console.log(sheetName)

app.get('/', (req, res) => {
  res.send('Api For Getting Data from Leet Code Website')
})

app.post('/api', async (req, res) => {
  const workbook = XLSX.readFile(
    'D://Clg/SEM6/CS312/Project/Code For Generating Dataset/Input/leetcode_indian_userrating.xlsx',
  )
  const sheetName = workbook.SheetNames[0]
  const worksheet = workbook.Sheets[sheetName]

  const newSheetName = 'New Sheet'
  const newSheetData = []
  var val = 0

  // Loop through rows in column B and get usernames
  for (let i = 1; i <= 14000; i++) {
    
    const cell = worksheet['B' + i]
    if (cell && cell.v) {
      console.log(cell.v)

      const query = {
        query:
          '\n      query languageStats($username: String!) {\n  matchedUser(username: $username) {\n    languageProblemCount {\n      languageName\n      problemsSolved\n    }\n  }\n}\n   ',
        variables: {
          username: `${cell.v}`,
        },
        operationName: 'languageStats',
      }


      try {
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
            username: `${cell.v}`,
          },
          operationName: 'languageStats',
        }

        const response = await axios.post(
          'https://leetcode.com/graphql/',
          query,
        )
        const languageProblemCounts = response.data.data.matchedUser.languageProblemCount.map(
          ({ languageName, problemsSolved }) => ({
            languageName,
            problemsSolved,
          }),
        )
        const data = {}
        languageProblemCounts.forEach(({ languageName, problemsSolved }) => {
          if (!data[languageName]) {
            data[languageName] = []
          }
          data[languageName].push(problemsSolved)
        })

        console.log(data)
        const arr = Object.entries(data)

        const data2 = {
          sno: val,
          username: cell.v,
          languageProblemCounts: arr,
        }
        console.log(data2)
        newSheetData.push(data2)
        val = val + 1

        // continue with your code here
      } catch (error) {
        console.error(error)
      }
    }
  }

  console.log(newSheetData)

  res.json(newSheetData)
})

app.listen(process.env.PORT || 3002, function () {
  console.log('Express app running on port ' + (process.env.PORT || 3002))
})
