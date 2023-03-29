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
    'D://Clg/SEM6/CS312/Project/Input/leetcode_indian_userrating.xlsx',
  )
  const sheetName = workbook.SheetNames[0]
  const worksheet = workbook.Sheets[sheetName]

  const newSheetName = 'New Sheet'
  const newSheetData = []
  var val = 0

  // Loop through rows in column B and get usernames
  for (let i = 1; i <= 16600; i++) {
    const cell = worksheet['B' + i]
    if (cell && cell.v) {
      console.log(cell.v)

      const query = {
        query:
          '\n    query userProblemsSolved($username: String!) {\n   matchedUser(username: $username) {\n    submitStatsGlobal {\n      acSubmissionNum {\n        difficulty\n        count\n      }\n    }\n  }\n}\n    ',
        variables: {
          username: `${cell.v}`,
        },
        operationName: 'userProblemsSolved',
      }

      const query2 = {
        query:
          '\n    query userPublicProfile($username: String!) {\n  matchedUser(username: $username) {\n  profile {\n      ranking\n       realName\n         countryName\n   postViewCount\n   solutionCount\n              reputation\n            }\n  }\n}\n    ',
        variables: {
          username: `${cell.v}`,
        },
        operationName: 'userPublicProfile',
      }
      const query3 = {
        query:
          '\n    query userProfileCalendar($username: String!, $year: Int) {\n  matchedUser(username: $username) {\n    userCalendar(year: $year) {\n      activeYears\n      streak\n      totalActiveDays\n            }\n  }\n}\n    ',
        variables: {
          username: `${cell.v}`,
        },
        operationName: 'userProfileCalendar',
      }

      const query4 = {
        query:
          '\n    query userContestRankingInfo($username: String!) {\n  userContestRanking(username: $username) {\n    attendedContestsCount\n    rating\n    globalRanking\n    totalParticipants\n    topPercentage\n    badge {\n      name\n    }\n  }\n}\n    ',
        variables: {
          username: `${cell.v}`,
        },
        operationName: 'userContestRankingInfo',
      }
      const apiUrl = 'https://leetcode.com/graphql/'
      var data1
      var data2
      var data3
      var data4
      await axios
        .post(apiUrl, query)
        .then(async (response) => {
          data1 =
            response.data.data.matchedUser.submitStatsGlobal.acSubmissionNum
          console.log(data1)
          await axios
            .post(apiUrl, query2)
            .then(async (response) => {
              data2 = data1.concat(response.data.data.matchedUser.profile)
              await axios
                .post(apiUrl, query3)
                .then(async (response) => {
                  data3 = data2.concat(
                    response.data.data.matchedUser.userCalendar,
                  )
                  await axios
                    .post(apiUrl, query4)
                    .then((response) => {
                      data4 = data3.concat(
                        response.data.data.userContestRanking,
                      )
                      val = val + 1

                      const Data = {
                        Sno: val,
                        Username: cell.v,
                        Total: data3[0].count ? data3[0].count : null,
                        Easy: data3[1].count ? data3[1].count : null,
                        Medium: data3[2].count ? data3[2].count : null,
                        Hard: data3[3].count ? data3[3].count : null,
                        Name: data3[4].realName ? data3[4].realName : null,
                        Country: data3[4].countryName
                          ? data3[4].countryName
                          : null,
                        PostViewCount: data3[4].postViewCount
                          ? data3[4].postViewCount
                          : null,
                        SolutionCount: data3[4].solutionCount
                          ? data3[4].solutionCount
                          : null,
                        Reputation: data3[4].reputation
                          ? data3[4].reputation
                          : null,
                        ActiveYears: data3[5].activeYears
                          ? data3[5].activeYears
                          : null,
                        Streak: data3[5].streak ? data3[5].streak : null,
                        TotalActiveDays: data3[5].totalActiveDays
                          ? data3[5].totalActiveDays
                          : null,
                        AttendedContestsCount: data4[6]
                          ? data4[6].attendedContestsCount
                            ? data4[6].attendedContestsCount
                            : null
                          : null,
                        Rating: data4[6]
                          ? data4[6].rating
                            ? data4[6].rating
                            : null
                          : null,
                        GlobalRanking: data4[6]
                          ? data4[6].globalRanking
                            ? data4[6].globalRanking
                            : null
                          : null,
                        TotalParticipants: data4[6]
                          ? data4[6].totalParticipants
                            ? data4[6].totalParticipants
                            : null
                          : null,
                        TopPercentage: data4[6]
                          ? data4[6].topPercentage
                            ? data4[6].topPercentage
                            : null
                          : null,
                        Badge: data4[6]
                          ? data4[6].badge
                            ? data4[6].badge.name
                            : null
                          : null,
                      }

                      if (newSheetData.includes(Data) === false) {
                        newSheetData.push(Data)
                      }

                      console.log(Data)
                      console.log(val)
                    })
                    .catch((error) => {
                      console.error(error)
                    })
                })
                .catch((error) => {
                  console.error(error)
                })
            })
            .catch((error) => {
              console.error(error)
            })
        })
        .catch((error) => {
          console.error(error)
          data1 = 0
        })
    }
  }

  console.log(newSheetData)
  const newSheetDataString = newSheetData.map((data) => ({
    ...data,
    ActiveYears: data.ActiveYears.join(', '),
  }))

  const newWorkbook = XLSX.utils.book_new()
  const newSheet = XLSX.utils.json_to_sheet(newSheetDataString)
  XLSX.utils.book_append_sheet(newWorkbook, newSheet, newSheetName)

  XLSX.writeFile(newWorkbook, 'D://Clg/SEM6/CS312/Project/Output/Dataset.xlsx')

  res.json(newSheetData)
})

app.listen(process.env.PORT || 3002, function () {
  console.log('Express app running on port ' + (process.env.PORT || 3002))
})
